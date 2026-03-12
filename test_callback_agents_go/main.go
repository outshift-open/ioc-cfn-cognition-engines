// test_callback_agents_go — Go replica of test_callback_agents.py
//
// Three NegMAS-style aspiration (Boulware) concession agents share a single
// HTTP server on port 8091.  They respond to batch callback messages sent by
// the semantic-negotiation-agent server (port 8089).  Compatible with the
// Python implementation: same aspiration formula, same SSTP envelope format,
// same trace file layout.
//
// Protocol types are imported from the local sstp/ sub-package, which mirrors
// the Python protocol/sstp/ module exactly.
//
// Usage:
//
//	# Terminal 1 — start the negotiation server:
//	cd semantic-negotiation-agent
//	poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089
//
//	# Terminal 2 — run this program:
//	cd test_callback_agents_go
//	go run . [--neg-server http://localhost:8089]
//
// Agents:
//
//	Agent A  prefer_low=true   exponent=1.5  min_reservation=0.2  (near-linear conceder, cheap)
//	Agent B  prefer_low=false  exponent=3.0  min_reservation=0.2  (hard Boulware, premium)
//	Agent C  prefer_low=true   exponent=2.0  min_reservation=0.1  (balanced conceder, cheap)
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"test_callback_agents_go/sstp"

	"gopkg.in/yaml.v3"
)

// ── Constants ──────────────────────────────────────────────────────────────

const (
	defaultNegServer = "http://localhost:8089"
	agentPort        = 8091
)

// ── Local file utilities ───────────────────────────────────────────────────

// saveJSON writes v as indented JSON to path, creating parent dirs as needed.
func saveJSON(path string, v any) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		log.Printf("saveJSON mkdir %s: %v", path, err)
		return
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		log.Printf("saveJSON marshal %s: %v", path, err)
		return
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		log.Printf("saveJSON write %s: %v", path, err)
	}
}

// imax returns the larger of two ints.
func imax(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ── Mission YAML loading ──────────────────────────────────────────────────

// Mission represents a single negotiation scenario loaded from missions.yaml.
// Issues and options are discovered dynamically by the semantic negotiation
// pipeline (Component 1: IntentDiscovery, Component 2: OptionsGeneration).
type Mission struct {
	Name        string `yaml:"name"`
	ContentText string `yaml:"content_text"`
	NSteps      int    `yaml:"n_steps"`
}

type missionsFile struct {
	Missions []Mission `yaml:"missions"`
}

// loadMissions reads a missions.yaml file and returns the mission list.
func loadMissions(path string) ([]Mission, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var mf missionsFile
	if err := yaml.Unmarshal(data, &mf); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(mf.Missions) == 0 {
		return nil, fmt.Errorf("%s contains no missions", path)
	}
	return mf.Missions, nil
}

// ── NegMAS concession agent ───────────────────────────────────────────────

// Agent implements the NegMAS AspirationNegotiator concession formula:
//
//	aspiration(t) = max(minReservation, 1 − t^exponent)
//
// where t = round / n_steps ∈ [0, 1].
//
// Preferences are derived lazily from the first payload that carries
// options_per_issue (the agent has no prior knowledge of the negotiation space).
type Agent struct {
	Name           string
	PreferLow      bool    // true → prefer index-0 options; false → prefer last-index options
	Exponent       float64 // concession exponent (>1 = Boulware, 1 = linear, <1 = conceder)
	MinReservation float64 // hard utility floor — never accept below this
	mu             sync.Mutex
	prefs          map[string]map[string]float64 // issue → {option → utility}; built lazily
}

// buildPrefs derives a utility map from wire-format options.
// For prefer_low:  option at index i → utility 1 − i/(n−1)  (best = index 0).
// For prefer_high: option at index i → utility i/(n−1)      (best = last index).
func (a *Agent) buildPrefs(optionsPerIssue map[string][]string) map[string]map[string]float64 {
	prefs := make(map[string]map[string]float64, len(optionsPerIssue))
	for issue, opts := range optionsPerIssue {
		n := len(opts)
		denom := float64(imax(n-1, 1))
		m := make(map[string]float64, n)
		for i, o := range opts {
			if a.PreferLow {
				m[o] = math.Round((1.0-float64(i)/denom)*1000) / 1000
			} else {
				m[o] = math.Round((float64(i)/denom)*1000) / 1000
			}
		}
		prefs[issue] = m
	}
	return prefs
}

func (a *Agent) ensurePrefs(optionsPerIssue map[string][]string) map[string]map[string]float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.prefs) != len(optionsPerIssue) {
		a.prefs = a.buildPrefs(optionsPerIssue)
	}
	return a.prefs
}

// resetPrefs clears cached preferences so the agent rebuilds them for the next mission.
func (a *Agent) resetPrefs() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.prefs = nil
}

// utility computes the mean utility for offer across all issues.
func (a *Agent) utility(offer map[string]string, optionsPerIssue map[string][]string) float64 {
	prefs := a.ensurePrefs(optionsPerIssue)
	var total float64
	count := 0
	for issue, val := range offer {
		if m, ok := prefs[issue]; ok {
			total += m[val]
			count++
		}
	}
	if count == 0 {
		return 0.0
	}
	return total / float64(count)
}

// aspiration returns max(minReservation, 1 − t^exponent).
func (a *Agent) aspiration(t float64) float64 {
	v := 1.0 - math.Pow(t, a.Exponent)
	if v < a.MinReservation {
		return a.MinReservation
	}
	return v
}

// outcomeEntry is a (offer, utility) pair used for sorting.
type outcomeEntry struct {
	offer   map[string]string
	utility float64
}

// allOutcomesSorted returns all possible outcomes sorted by utility descending.
// Issues are enumerated in alphabetical order (consistent with Python's dict
// order-agnostic behaviour — utility is a simple mean so order doesn't matter).
func (a *Agent) allOutcomesSorted(optionsPerIssue map[string][]string) []outcomeEntry {
	prefs := a.ensurePrefs(optionsPerIssue)
	issues := make([]string, 0, len(optionsPerIssue))
	for issue := range optionsPerIssue {
		issues = append(issues, issue)
	}
	sort.Strings(issues) // deterministic ordering
	var results []outcomeEntry
	var recurse func(idx int, current map[string]string)
	recurse = func(idx int, current map[string]string) {
		if idx == len(issues) {
			offer := make(map[string]string, len(current))
			for k, v := range current {
				offer[k] = v
			}
			var total float64
			for _, issue := range issues {
				total += prefs[issue][current[issue]]
			}
			u := math.Round(total/float64(len(issues))*10000) / 10000
			results = append(results, outcomeEntry{offer: offer, utility: u})
			return
		}
		issue := issues[idx]
		for _, opt := range optionsPerIssue[issue] {
			current[issue] = opt
			recurse(idx+1, current)
		}
	}
	recurse(0, make(map[string]string, len(issues)))
	sort.Slice(results, func(i, j int) bool {
		return results[i].utility > results[j].utility
	})
	return results
}

// decidePropose returns the minimum-utility outcome that still satisfies
// aspiration(t), falling back to the ideal (maximum-utility) outcome.
// This replicates Python's NegMASConcessionAgent.decide_propose exactly.
func (a *Agent) decidePropose(roundNum, nSteps int, optionsPerIssue map[string][]string) (map[string]string, float64) {
	t := float64(roundNum) / float64(imax(nSteps, 1))
	asp := a.aspiration(t)
	outcomes := a.allOutcomesSorted(optionsPerIssue)
	best := outcomes[0].offer // fallback: ideal offer
	for _, e := range outcomes {
		if e.utility >= asp {
			best = e.offer // keep updating → minimum qualifying outcome
		} else {
			break // sorted desc — no further outcome qualifies
		}
	}
	return best, asp
}

// decideRespond returns "accept" if utility(offer) ≥ aspiration(t), else "reject".
func (a *Agent) decideRespond(offer map[string]string, roundNum, nSteps int, optionsPerIssue map[string][]string) string {
	t := float64(roundNum) / float64(imax(nSteps, 1))
	asp := a.aspiration(t)
	u := a.utility(offer, optionsPerIssue)
	decision := "reject"
	if u >= asp {
		decision = "accept"
	}
	fmt.Printf("  [%s] respond  round=%d  utility=%.3f  aspiration=%.3f  → %s\n",
		a.Name, roundNum, u, asp, decision)
	return decision
}

// ── HTTP handler ──────────────────────────────────────────────────────────

// makeHandler returns an http.HandlerFunc for POST /decide.
// It receives []sstp.NegotiateMessage (one per participant), processes each
// message synchronously, POSTs the full reply list to
// {negServer}/api/v1/negotiate/agents-decisions, and only then returns
// {"status":"ack"}.  This mirrors the Python test-agent behaviour and
// guarantees that by the time the ACK reaches BatchCallbackRunner the
// decisions are already in the server's _DECISIONS store.
//
// traceDir is a pointer so the caller can update it between missions.
func makeHandler(agents map[string]*Agent, traceDir *string, negServer string, client *http.Client) http.HandlerFunc {
	// Pick any agent as fallback.
	var fallback *Agent
	for _, a := range agents {
		fallback = a
		break
	}
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read error", http.StatusBadRequest)
			return
		}
		// Parse as array of raw JSON messages so we can unmarshal each independently.
		var rawMessages []json.RawMessage
		if err := json.Unmarshal(body, &rawMessages); err != nil {
			http.Error(w, "json error: "+err.Error(), http.StatusBadRequest)
			return
		}
		replies := make([]sstp.NegotiateMessage, 0, len(rawMessages))
		for _, rawMsg := range rawMessages {
			var msg sstp.NegotiateMessage
			if err := json.Unmarshal(rawMsg, &msg); err != nil {
				log.Printf("unmarshal NegotiateMessage: %v", err)
				continue
			}
			payload := msg.Payload
			// Dispatch to the right agent by participant_id.
			participantID, _ := payload["participant_id"].(string)
			agent := agents[participantID]
			// Fuzzy match if exact key not found.
			if agent == nil {
				for pid, a := range agents {
					if strings.Contains(pid, sstp.Slug(participantID)) ||
						strings.Contains(sstp.Slug(participantID), pid) {
						agent = a
						break
					}
				}
			}
			if agent == nil {
				agent = fallback
			}
			// Extract fields from payload.
			action, _ := payload["action"].(string)
			roundNumF, _ := payload["round"].(float64)
			roundNum := int(roundNumF)
			if roundNum == 0 {
				roundNum = 1
			}
			nStepsF, _ := payload["n_steps"].(float64)
			nSteps := int(nStepsF)
			if nSteps == 0 {
				nSteps = 200
			}
			issues := extractStringSlice(payload["issues"])
			optionsPerIssue := extractStringSliceMap(payload["options_per_issue"])
			sessionID := msg.SemanticContext.SessionID
			if sessionID == "" {
				sessionID = "unknown-session"
			}
			// Capture raw SAOState bytes for verbatim echo-back.
			// The server verifies a SHA-256 checksum over these bytes.
			saoStateRaw := msg.SemanticContext.SAOState
			if len(saoStateRaw) == 0 {
				saoStateRaw = json.RawMessage("null")
			}
			isShadow, _ := payload["is_shadow_call"].(bool)
			agentSlg := sstp.Slug(agent.Name)
			roundDir := filepath.Join(*traceDir, fmt.Sprintf("round_%04d", roundNum))
			// Save incoming request to trace.
			if !isShadow {
				var rawObj any
				if err := json.Unmarshal(rawMsg, &rawObj); err == nil {
					saveJSON(filepath.Join(roundDir,
						fmt.Sprintf("%s__%s__request.json", action, agentSlg)), rawObj)
				}
			}
			// Decide and build the sstp.NegotiateMessage reply.
			var reply sstp.NegotiateMessage
			switch action {
			case "propose":
				offer, asp := agent.decidePropose(roundNum, nSteps, optionsPerIssue)
				if !isShadow {
					fmt.Printf("  [%s] propose  round=%d  aspiration=%.3f  offer=%v\n",
						agent.Name, roundNum, asp, offer)
				}
				offerAny := make(map[string]any, len(offer))
				for k, v := range offer {
					offerAny[k] = v
				}
				replyPayload := map[string]any{
					"action":            "counter_offer",
					"round":             roundNum,
					"issues":            issues,
					"options_per_issue": optionsPerIssue,
					"offer":             offer,
				}
				saoResp := &sstp.SAOResponse{
					Response: sstp.ResponseRejectOffer,
					Outcome:  sstp.OutcomeFromMap(offerAny),
				}
				reply = sstp.BuildReply(sessionID, agent.Name, replyPayload, saoResp, saoStateRaw)
			case "respond":
				currentOffer := extractCurrentOffer(payload["current_offer"])
				decision := agent.decideRespond(currentOffer, roundNum, nSteps, optionsPerIssue)
				var saoResp *sstp.SAOResponse
				if decision == "accept" {
					outcomeAny := make(map[string]any, len(currentOffer))
					for k, v := range currentOffer {
						outcomeAny[k] = v
					}
					saoResp = &sstp.SAOResponse{
						Response: sstp.ResponseAcceptOffer,
						Outcome:  sstp.OutcomeFromMap(outcomeAny),
					}
				} else {
					saoResp = &sstp.SAOResponse{Response: sstp.ResponseRejectOffer}
				}
				replyPayload := map[string]any{
					"action":            decision,
					"round":             roundNum,
					"issues":            issues,
					"options_per_issue": optionsPerIssue,
				}
				reply = sstp.BuildReply(sessionID, agent.Name, replyPayload, saoResp, saoStateRaw)
			default:
				replyPayload := map[string]any{"action": "reject", "round": roundNum}
				saoResp := &sstp.SAOResponse{Response: sstp.ResponseRejectOffer}
				reply = sstp.BuildReply(sessionID, agent.Name, replyPayload, saoResp, saoStateRaw)
			}
			// Save reply to trace.
			if !isShadow {
				saveJSON(filepath.Join(roundDir,
					fmt.Sprintf("%s__%s__reply.json", action, agentSlg)), reply)
			}
			replies = append(replies, reply)
		}
		// Synchronously POST decisions to the negotiation server before returning
		// the ACK.  BatchCallbackRunner pops from _DECISIONS immediately after
		// reading the ACK — the causal order guarantees no race condition.
		repliesJSON, err := json.Marshal(replies)
		if err != nil {
			log.Printf("/decide: marshal replies: %v", err)
			http.Error(w, "marshal error", http.StatusInternalServerError)
			return
		}
		decisionsURL := negServer + "/api/v1/negotiate/agents-decisions"
		dResp, err := client.Post(decisionsURL, "application/json", bytes.NewReader(repliesJSON))
		if err != nil {
			log.Printf("/decide: POST agents-decisions failed: %v", err)
			http.Error(w, "upstream error", http.StatusBadGateway)
			return
		}
		dResp.Body.Close()
		if dResp.StatusCode >= 300 {
			log.Printf("/decide: agents-decisions returned HTTP %d", dResp.StatusCode)
			http.Error(w, "upstream status", http.StatusBadGateway)
			return
		}

		// Decisions are now stored server-side — return the ACK.
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ack"}`)
	}
}

// extractStringSlice safely converts []interface{} → []string.
func extractStringSlice(v any) []string {
	arr, ok := v.([]interface{})
	if !ok {
		return []string{}
	}
	out := make([]string, 0, len(arr))
	for _, x := range arr {
		if s, ok := x.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

// extractStringSliceMap safely converts map[string]interface{} → map[string][]string.
func extractStringSliceMap(v any) map[string][]string {
	m, ok := v.(map[string]interface{})
	if !ok {
		return map[string][]string{}
	}
	out := make(map[string][]string, len(m))
	for issue, vals := range m {
		out[issue] = extractStringSlice(vals)
	}
	return out
}

// extractCurrentOffer safely converts map[string]interface{} → map[string]string.
func extractCurrentOffer(v any) map[string]string {
	m, ok := v.(map[string]interface{})
	if !ok {
		return map[string]string{}
	}
	out := make(map[string]string, len(m))
	for k, val := range m {
		if s, ok := val.(string); ok {
			out[k] = s
		}
	}
	return out
}

// ── Server readiness poll ─────────────────────────────────────────────────

func waitForServer(baseURL string, retries int, delay time.Duration) error {
	client := &http.Client{Timeout: 2 * time.Second}
	for i := 0; i < retries; i++ {
		resp, err := client.Get(baseURL + "/openapi.json")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}
		time.Sleep(delay)
	}
	return fmt.Errorf("server at %s did not become ready", baseURL)
}

// ── main ──────────────────────────────────────────────────────────────────

func main() {
	negServerFlag := flag.String("neg-server", defaultNegServer,
		"Base URL of the semantic-negotiation-agent server")
	traceDirFlag := flag.String("trace-dir", "neg_trace_go",
		"Root directory for per-run trace folders (each run creates a timestamped sub-dir)")
	defaultMissionsFile := filepath.Join("..", "missions.yaml")
	missionsFileFlag := flag.String("missions-file", defaultMissionsFile,
		"Path to missions YAML file")
	flag.Parse()
	negServer := *negServerFlag

	// ── trace root (timestamped) ─────────────────────────────────────────────
	timestamp := time.Now().Format("20060102_150405")
	traceRoot := filepath.Join(*traceDirFlag, timestamp)
	if err := os.MkdirAll(traceRoot, 0o755); err != nil {
		log.Fatalf("create trace root: %v", err)
	}
	fmt.Printf("Run trace root: %s\n", traceRoot)

	// ── load missions ────────────────────────────────────────────────────────
	missionsPath, err := filepath.Abs(*missionsFileFlag)
	if err != nil {
		log.Fatalf("resolve missions path: %v", err)
	}
	missions, err := loadMissions(missionsPath)
	if err != nil {
		log.Fatalf("load missions: %v", err)
	}

	fmt.Printf("Missions file : %s\n", missionsPath)
	fmt.Printf("Missions to negotiate: %d\n", len(missions))
	for _, m := range missions {
		fmt.Printf("  • %s\n", m.Name)
	}
	fmt.Println()

	// ── three agents (same config as Python test) ─────────────────────────────
	agents := map[string]*Agent{
		"agent-a": {Name: "Agent A", PreferLow: true, Exponent: 1.5, MinReservation: 0.2},
		"agent-b": {Name: "Agent B", PreferLow: false, Exponent: 3.0, MinReservation: 0.2},
		"agent-c": {Name: "Agent C", PreferLow: true, Exponent: 2.0, MinReservation: 0.1},
	}

	// ── start agent HTTP server ───────────────────────────────────────────────
	// httpClient is shared by both the agent handler (to POST agents-decisions)
	// and the mission loop (to POST initiate).
	httpClient := &http.Client{Timeout: 120 * time.Second}

	// traceDir is a pointer updated per-mission so the handler writes to the right folder.
	currentTraceDir := traceRoot
	mux := http.NewServeMux()
	mux.HandleFunc("/decide", makeHandler(agents, &currentTraceDir, negServer, httpClient))
	// Minimal /openapi.json so waitForServer can health-check.
	mux.HandleFunc("/openapi.json", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"info":{"title":"Go Agent Server","version":"1.0"}}`)
	})
	agentSrv := &http.Server{
		Addr:    fmt.Sprintf(":%d", agentPort),
		Handler: mux,
	}
	go func() {
		if err := agentSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("agent server: %v", err)
		}
	}()
	fmt.Printf("Starting shared agent server on :%d…\n", agentPort)
	if err := waitForServer(fmt.Sprintf("http://localhost:%d", agentPort), 20, 300*time.Millisecond); err != nil {
		log.Fatalf("agent server did not start: %v", err)
	}
	fmt.Println("Agent server is up.")
	fmt.Println()

	// ── verify negotiation server is reachable ────────────────────────────────
	{
		client := &http.Client{Timeout: 5 * time.Second}
		resp, err := client.Get(negServer + "/openapi.json")
		if err != nil || resp.StatusCode >= 500 {
			if resp != nil {
				resp.Body.Close()
			}
			fmt.Printf("ERROR: negotiation server at %s is not reachable: %v\n", negServer, err)
			fmt.Println("Start it with:  cd semantic-negotiation-agent && " +
				"poetry run uvicorn app.main:app --host 0.0.0.0 --port 8089")
			os.Exit(1)
		}
		resp.Body.Close()
	}

	// ── multi-mission loop ────────────────────────────────────────────────────
	agentCallbackURL := fmt.Sprintf("http://localhost:%d/decide", agentPort)

	for idx, mission := range missions {
		// Reset cached preferences — issue space may differ between missions.
		for _, ag := range agents {
			ag.resetPrefs()
		}

		missionSlug := sstp.Slug(mission.Name)
		missionTraceDir := filepath.Join(traceRoot, missionSlug)
		if err := os.MkdirAll(missionTraceDir, 0o755); err != nil {
			log.Fatalf("create mission trace dir: %v", err)
		}
		// Point the running agent server at this mission's trace folder.
		currentTraceDir = missionTraceDir

		sessionID := fmt.Sprintf("sess-%s-%s", timestamp, missionSlug)

		fmt.Printf("%s\n", strings.Repeat("=", 62))
		fmt.Printf("  Mission %d/%d: %s\n", idx+1, len(missions), mission.Name)
		fmt.Printf("  Trace  : %s\n", missionTraceDir)
		fmt.Printf("%s\n\n", strings.Repeat("=", 62))

		// Build initiate payload — issues/options discovered dynamically by the server.
		innerPayload := map[string]any{
			"content_text": mission.ContentText,
			"agents": []map[string]any{
				{"id": "agent-a", "name": "Agent A", "callback_url": agentCallbackURL},
				{"id": "agent-b", "name": "Agent B", "callback_url": agentCallbackURL},
				{"id": "agent-c", "name": "Agent C", "callback_url": agentCallbackURL},
			},
			"n_steps": mission.NSteps,
		}
		sc := sstp.DefaultNegotiateSemanticContext(sessionID)
		initMsg := sstp.DefaultNegotiateMessage(
			"init-"+sessionID,
			time.Now().UTC().Format(time.RFC3339),
			sstp.Origin{ActorID: "test-runner-go", TenantID: "demo"},
			sc,
		)
		initMsg.Payload = innerPayload
		initMsg.PayloadHash = strings.Repeat("0", 64)
		saveJSON(filepath.Join(missionTraceDir, "00_initiate_request.json"), initMsg)

		fmt.Printf("POST %s/api/v1/negotiate/initiate …\n", negServer)
		initBody, err := json.Marshal(initMsg)
		if err != nil {
			log.Fatalf("marshal initiate: %v", err)
		}
		httpResp, err := httpClient.Post(
			negServer+"/api/v1/negotiate/initiate",
			"application/json",
			bytes.NewReader(initBody),
		)
		if err != nil {
			log.Fatalf("initiate POST: %v", err)
		}
		fmt.Printf("HTTP %d\n", httpResp.StatusCode)
		respBody, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		var result any
		if err := json.Unmarshal(respBody, &result); err == nil {
			saveJSON(filepath.Join(missionTraceDir, "final_result.json"), result)
			pretty, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(pretty))
		} else {
			fmt.Println(string(respBody))
		}
		fmt.Printf("\nMission %d trace saved to: %s\n\n", idx+1, missionTraceDir)
	}

	fmt.Printf("All %d missions complete.  Run trace root: %s\n", len(missions), traceRoot)
}
