from typing import Dict, List, Tuple
import asyncio
import json
import os
from dotenv import find_dotenv

# Global counter of successful LLM chat calls (Azure requests that returned)
_LLM_CALL_COUNT = 0


def get_llm_call_count() -> int:
    return _LLM_CALL_COUNT


def _inc_llm_call_count() -> None:
    global _LLM_CALL_COUNT
    _LLM_CALL_COUNT += 1


class _LLMBaseClient:
    """
    Shared Azure OpenAI client setup and utilities.
    Subclasses should use _call_chat(...) and check azure availability via self._azure_client.
    """

    def __init__(self, temperature: float, client_label: str):
        self.temperature = temperature
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        # API version is optional; if not provided, use an internal safe default
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self._azure_client = None
        # Log basic env discovery (masked)
        env_path = find_dotenv()
        masked = None
        if self.api_key:
            masked = (self.api_key[:2] + "..." + self.api_key[-4:]) if len(self.api_key) > 6 else "***"
        print(
            f"[{client_label}] init | dotenv='{env_path or '(none)'}' | "
            f"endpoint='{self.endpoint or '(missing)'}' | deployment='{self.deployment or '(missing)'}' | "
            f"api_key='{masked or '(missing)'}'"
        )
        if self.endpoint and self.api_key and self.deployment:
            try:
                from openai import AzureOpenAI

                # Provide a default API version only if none was supplied via env
                _effective_api_version = self.api_version or "2024-06-01"
                self._azure_client = AzureOpenAI(
                    api_key=self.api_key, api_version=_effective_api_version, azure_endpoint=self.endpoint
                )
                print(f"[{client_label}] Azure configured: " f"deployment='{self.deployment}'")
            except Exception:
                self._azure_client = None
        if not self._azure_client:
            print(f"[{client_label}] Azure not configured; using fallback.")

    def _call_chat(self, system: str, user: str) -> str:
        """
        Invoke Azure OpenAI chat completion, returning the message content (may be '').
        Raises if client not configured or request fails.
        """
        if not self._azure_client:
            raise RuntimeError("Azure client not configured")
        resp = self._azure_client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.temperature,
        )
        # Count only successful upstream LLM calls
        _inc_llm_call_count()
        return (resp.choices[0].message.content or "").strip()


class EvidenceJudge(_LLMBaseClient):
    """
    LLM client for:
    - Selecting most relevant paths and declaring sufficiency
    Uses Azure OpenAI if configured; otherwise, provides deterministic fallbacks.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="EvidenceJudge")

    def select_paths_and_check_sufficiency(
        self, question: str, candidate_paths: List[str], select_k: int
    ) -> Tuple[List[int], bool, str]:
        if not candidate_paths:
            return [], False, ""
        if not self._azure_client:
            k = min(select_k, len(candidate_paths))
            selected = list(range(k))
            print(f"[EvidenceJudge] Fallback select: {selected} | sufficient=False")
            return selected, False, "fallback: azure not configured"

        system = (
            "You are an evidence-based reasoning judge selecting the most relevant knowledge paths to answer a query. "
            "Your task is NOT keyword matching, but logical evaluation of whether each path provides evidence "
            "for answering the question.\n\n"
            " Respond with STRICT JSON ONLY. Follow these rules exactly:\n"
            "- Output a SINGLE JSON object and NOTHING ELSE (no code fences, no prose):\n"
            '  {"selected": [indices], "sufficient": true|false, "reason": "<one-line>"}\n'
            '- "selected" is an array of 0-based integers that refer to the shown candidates.\n'
            '- "reason" MUST be a concise single sentence (max ~20 words) justifying the sufficiency decision.\n'
            "- Do not include trailing commas, comments, or any extra fields."
        )
        numbered = "\n".join([f"{i}. {p}" for i, p in enumerate(candidate_paths)])
        user = f"Question: {question or '(none)'}\n\nCandidate paths:\n{numbered}\n\nSelect top {select_k} paths."

        try:
            # Log what the LLM is given (selection)
            try:
                print(f"[EvidenceJudge][Select] candidates={len(candidate_paths)}")
                preview_count = min(10, len(candidate_paths))
                for i in range(preview_count):
                    snippet = (candidate_paths[i] or "")[:200].replace("\n", " ")
                    print(f"[EvidenceJudge][Select]   [{i}] {snippet}")
                if len(candidate_paths) > preview_count:
                    print(f"[EvidenceJudge][Select]   ... {len(candidate_paths) - preview_count} more not shown")
            except Exception:
                pass
            content = self._call_chat(system, user)
            data = json.loads(content)
            selected = data.get("selected") or []
            sufficient = bool(data.get("sufficient", False))
            reason_raw = data.get("reason") or ""
            # Keep a single-line, trimmed reason
            reason = (str(reason_raw).splitlines()[0] if isinstance(reason_raw, str) else "").strip()
            clean = [i for i in selected if isinstance(i, int) and 0 <= i < len(candidate_paths)]
            clean = clean[:select_k]
            print(f"[EvidenceJudge] Selected={clean} | sufficient={sufficient} | reason='{reason}'")
            # Log selected snippets
            try:
                for i in clean:
                    if 0 <= i < len(candidate_paths):
                        snippet = (candidate_paths[i] or "")[:200].replace("\n", " ")
                        print(f"[EvidenceJudge][Select]   pick[{i}] {snippet}")
            except Exception:
                pass
            return clean, sufficient, reason
        except Exception:
            print("[EvidenceJudge] Azure call failed; falling back to heuristic selection.")
            k = min(select_k, len(candidate_paths))
            return list(range(k)), False, "fallback: azure error"


    # Async wrappers to avoid blocking the event loop with sync SDK calls
    async def async_select_paths_and_check_sufficiency(
        self, question: str, candidate_paths: List[str], select_k: int
    ) -> Tuple[List[int], bool, str]:
        return await asyncio.to_thread(self.select_paths_and_check_sufficiency, question, candidate_paths, select_k)

    # Note: Ranking is handled by EvidenceRanker; EvidenceJudge provides only selection/sufficiency.


class EvidenceRanker(_LLMBaseClient):
    """
    LLM client dedicated to ranking paths by importance in [0, 1].
    Uses Azure OpenAI if configured; otherwise, provides deterministic fallbacks.
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="EvidenceRanker")

    def rank_paths(self, question: str, candidate_paths_repr: List[str]) -> Dict[int, float]:
        if not candidate_paths_repr:
            return {}
        if not self._azure_client:
            n = len(candidate_paths_repr)
            if n == 1:
                return {0: 1.0}
            scores = {i: 1.0 - 0.5 * (i / (n - 1)) for i in range(n)}
            print(f"[EvidenceRanker] Fallback rank produced {len(scores)} scores.")
            return scores

        system = (
            "You are ranking knowledge paths by how much they contribute to answering a question. "
            "Score each candidate on a 0.0 to 1.0 scale (1.0 = highly useful, 0.0 = irrelevant). "
            "Respond with STRICT JSON ONLY. Follow these rules exactly:\n"
            "- Output a SINGLE JSON object and NOTHING ELSE (no code fences, no prose):\n"
            '  {"scores": [{"index": i, "score": number}]}\n'
            '- Each "index" is a 0-based integer for a shown candidate.\n'
            '- Each "score" MUST be a number in [0, 1].\n'
            "- Do not include trailing commas, comments, or any extra fields."
        )
        numbered = "\n".join([f"{i}. {p}" for i, p in enumerate(candidate_paths_repr)])
        user = f"Question: {question or '(none)'}\n\nCandidate paths:\n{numbered}\n\nRank all items."

        try:
            try:
                print(f"[EvidenceRanker][Rank] candidates={len(candidate_paths_repr)}")
                preview_count = min(10, len(candidate_paths_repr))
                for i in range(preview_count):
                    snippet = (candidate_paths_repr[i] or "")[:200].replace("\n", " ")
                    print(f"[EvidenceRanker][Rank]   [{i}] {snippet}")
                if len(candidate_paths_repr) > preview_count:
                    print(f"[EvidenceRanker][Rank]   ... {len(candidate_paths_repr) - preview_count} more not shown")
            except Exception:
                pass
            content = self._call_chat(system, user)
            data = json.loads(content)
            scores_list = data.get("scores") or []
            scores: Dict[int, float] = {}
            for item in scores_list:
                try:
                    idx = int(item.get("index"))
                    sc = float(item.get("score"))
                except Exception:
                    continue
                if 0 <= idx < len(candidate_paths_repr):
                    sc = 0.0 if sc < 0.0 else (1.0 if sc > 1.0 else sc)
                    scores[idx] = sc
            if not scores:
                n = len(candidate_paths_repr)
                if n == 1:
                    scores = {0: 1.0}
                else:
                    scores = {i: 1.0 - 0.5 * (i / (n - 1)) for i in range(n)}
            print(f"[EvidenceRanker] Ranked {len(scores)} candidates.")
            try:
                for i, sc in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: min(10, len(scores))]:
                    snippet = (candidate_paths_repr[i] or "")[:200].replace("\n", " ")
                    print(f"[EvidenceRanker][Rank]   score[{i}]={sc:.3f} | {snippet}")
            except Exception:
                pass
            return scores
        except Exception:
            print("[EvidenceRanker] Azure ranking failed; using fallback ranking.")
            n = len(candidate_paths_repr)
            if n == 1:
                return {0: 1.0}
            return {i: 1.0 - 0.5 * (i / (n - 1)) for i in range(n)}

    async def async_rank_paths(self, question: str, candidate_paths_repr: List[str]) -> Dict[int, float]:
        return await asyncio.to_thread(self.rank_paths, question, candidate_paths_repr)


class EntityExtractor(_LLMBaseClient):
    """
    LLM client for extracting entities from a ReasonerCognitionRequest using a strict JSON system prompt.
    Provides sync and async entrypoints with a deterministic fallback.
    """

    SYSTEM_PROMPT = (
        "You extract salient entities (proper nouns, products, APIs, teams, systems, key technical terms) "
        "from the user's intent and any provided text.\n"
        "Respond with STRICT JSON ONLY. Follow these rules exactly:\n"
        "- Output a SINGLE JSON object and NOTHING ELSE (no code fences, no prose):\n"
        '  {"entities": [{"name": "<entity-1>"}, {"name": "<entity-2>"}, ...]}\n'
        '- Each entity object MUST include a non-empty "name" string.\n'
        "- Do not include trailing commas, comments, or any extra fields."
    )

    STOPWORDS = {
        "what",
        "does",
        "do",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "with",
        "is",
        "are",
    }

    def __init__(self, temperature: float = 0.1):
        super().__init__(temperature=temperature, client_label="EntityExtractor")

    def _fallback_extract(self, intent: str, texts: List[str]) -> List[Dict]:
        import re

        combined = f"{intent} {' '.join(texts)}".lower()
        candidates = re.findall(r"[a-z][a-z0-9_-]{3,}", combined)
        keywords = []
        seen = set()
        for w in candidates:
            if w in self.STOPWORDS or w in seen:
                continue
            seen.add(w)
            keywords.append({"name": w})
            if len(keywords) >= 10:
                break
        out = keywords or ([{"name": intent.strip()}] if intent.strip() else [])
        print(f"[EntityExtractor] Fallback used. Entities={len(out)} Preview={[e['name'] for e in out[:5]]}")
        return out

    def extract_entities_from_request(self, request) -> List[Dict]:
        # Lazy import type to avoid circulars in static contexts
        intent = request.intent or ""
        texts: List[str] = []
        for rec in request.records or []:
            try:
                rt = rec.record_type.value if hasattr(rec.record_type, "value") else str(rec.record_type)
            except Exception:
                rt = str(rec.record_type)
            if rt == "string" and isinstance(rec.content, str):
                texts.append(rec.content)
            elif rt == "json":
                texts.append(json.dumps(rec.content, ensure_ascii=False, separators=(",", ":")))

        if not self._azure_client:
            return self._fallback_extract(intent, texts)

        user_prompt = f"INTENT:\n{intent}\n\nTEXT:\n" + ("\n".join(texts) if texts else "(none)")
        try:
            content = self._call_chat(self.SYSTEM_PROMPT, user_prompt)
        except Exception:
            return self._fallback_extract(intent, texts)
        print(f"[EntityExtractor] LLM content (trunc): {content[:180]}{'...' if len(content) > 180 else ''}")
        try:
            data = json.loads(content)
            ents = data.get("entities") if isinstance(data, dict) else data
            out = [{"name": e.get("name", "").strip()} for e in ents if isinstance(e, dict) and e.get("name")]
            print(f"[EntityExtractor] LLM extracted entities: {len(out)}")
            return out
        except Exception:
            print("[EntityExtractor] JSON parse failed; falling back.")
            return self._fallback_extract(intent, texts)

    async def async_extract_entities_from_request(self, request) -> List[Dict]:
        return await asyncio.to_thread(self.extract_entities_from_request, request)


class QueryDecomposer(_LLMBaseClient):
    """
    LLM client for decomposing a query into numbered, atomic statements with up to two entities.
    Output is parsed into a structured list for downstream use.
    """

    SYSTEM_PROMPT = (
        "You will receive a multi-hop question, which is composed of several interconnected queries, along"
        " with a list of topic entities that serve as the main keywords for the question. Your task is to break the"
        " question into simpler parts, using each topic entity once or twice(source and target) and provide a Chain of Thought (CoT) that"
        " shows how the topic entities are related. Note: Each simpler question should explore how"
        " one query connects to others or the answer. The goal is to systematically address each entity to derive"
        " the final answer.\n\n"
        "OUTPUT FORMAT (strict):\n"
        "- Generate lines in the exact form: #(number). (query) , ##entity1##entity2##\n"
        "- The entity set must list every entity explicitly referenced by the sentence (no omissions), up to two.\n"
        "- Do not include any extra text outside the numbered lines."
    )

    def __init__(self, temperature: float = 0.2):
        super().__init__(temperature=temperature, client_label="QueryDecomposer")

    def decompose(self, text: str, entities: List[str] | None = None) -> List[Dict]:
        """
        Returns: List[{index:int, sentence:str, entities:[str]}]
        """
        if not (text or "").strip():
            return []
        if not self._azure_client:
            # Fallback: return the original as a single-item decomposition with naive entity guess
            base = (text or "").strip()
            return [{"index": 1, "sentence": base, "entities": []}]
        try:
            # Include extracted entities both in system content and user content to guide decomposition
            safe_ents = [str(e).strip() for e in (entities or []) if str(e).strip()]
            system_content = self.SYSTEM_PROMPT
            if safe_ents:
                system_content = (
                    self.SYSTEM_PROMPT
                    + "\n\nTopic entities (use at most two per statement; avoid reuse):\n- "
                    + "\n- ".join(safe_ents)
                )
            user_input = f"Sentence:\n{text}".strip()
            content = self._call_chat(system_content, user_input)
        except Exception:
            base = (text or "").strip()
            return [{"index": 1, "sentence": base, "entities": []}]
        # Parse lines starting with '#'
        out: List[Dict] = []
        try:
            for line in (content or "").splitlines():
                line = line.strip()
                if not line or not line.startswith("#"):
                    continue
                # Expected pattern: #number. sentence , ##entity1##entity2
                # Be tolerant to spaces and missing punctuation
                idx = None
                try:
                    # extract leading integer after '#'
                    after_hash = line[1:]
                    num_str = ""
                    for ch in after_hash:
                        if ch.isdigit():
                            num_str += ch
                        else:
                            break
                    idx = int(num_str) if num_str else None
                except Exception:
                    idx = None
                # Extract entities between '##'
                parsed_entities: List[str] = []
                parts = line.split("##")
                if len(parts) >= 2:
                    # entities appear between ## ... ## ... ##
                    for i in range(1, len(parts), 2):
                        ent = parts[i].strip()
                        if ent:
                            parsed_entities.append(ent)
                # Extract sentence before first '##' (strip leading '#n.' if present)
                sent_part = parts[0]
                # remove leading '#n.' or '#n)'
                try:
                    # find first '.' after '#digits'
                    dot_pos = sent_part.find(".")
                    if dot_pos != -1:
                        sent_part = sent_part[dot_pos + 1 :].strip(" ,")
                    else:
                        # remove up to first space
                        space_pos = sent_part.find(" ")
                        if space_pos != -1:
                            sent_part = sent_part[space_pos + 1 :].strip(" ,")
                except Exception:
                    pass
                # Post-process enforcement: include supplied topic entities that appear in the sentence (max 2, ordered)
                final_entities: List[str] = parsed_entities[:2]
                try:
                    if safe_ents:
                        lower_sent = sent_part.lower()
                        located = []
                        for e in safe_ents:
                            el = e.lower()
                            pos = lower_sent.find(el)
                            if pos != -1:
                                located.append((pos, e))
                        if located:
                            located.sort(key=lambda t: t[0])
                            seen = set()
                            ordered = []
                            for _, e in located:
                                if e not in seen:
                                    seen.add(e)
                                    ordered.append(e)
                            final_entities = ordered[:2]
                except Exception:
                    pass
                out.append(
                    {
                        "index": idx if isinstance(idx, int) else (len(out) + 1),
                        "sentence": sent_part.strip(),
                        "entities": final_entities,
                    }
                )
        except Exception:
            out = [{"index": 1, "sentence": (text or "").strip(), "entities": []}]
        # Fallback if no valid lines parsed
        if not out:
            base = (text or "").strip()
            return [{"index": 1, "sentence": base, "entities": safe_ents[:2]}]
        return out

    async def async_decompose(self, text: str, entities: List[str] | None = None) -> List[Dict]:
        return await asyncio.to_thread(self.decompose, text, entities)
