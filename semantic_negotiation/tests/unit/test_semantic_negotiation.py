# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NegotiationModel (component 3 of the semantic negotiation pipeline)."""
import pytest

from app.agent.negotiation_model import (
    NegotiationModel,
    NegotiationOutcome,
    NegotiationParticipant,
    NegotiationResult,
)


class TestNegotiationModel:
    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------

    def test_instantiates_with_default_steps(self):
        model = NegotiationModel()
        assert model.n_steps == 100

    def test_instantiates_with_custom_steps(self):
        model = NegotiationModel(n_steps=42)
        assert model.n_steps == 42

    def test_invalid_n_steps_raises(self):
        with pytest.raises(ValueError):
            NegotiationModel(n_steps=0)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_single_participant_raises(
        self, model, issues, options_per_issue, two_participants
    ):
        with pytest.raises(ValueError, match="two participants"):
            model.run(issues, options_per_issue, two_participants[:1])

    def test_missing_issue_in_options_raises(
        self, model, options_per_issue, two_participants
    ):
        with pytest.raises(ValueError, match="no entry"):
            model.run(["budget", "nonexistent"], options_per_issue, two_participants)

    def test_empty_options_list_raises(self, model, two_participants):
        with pytest.raises(ValueError, match="empty options"):
            model.run(["budget"], {"budget": []}, two_participants)

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_run_returns_negotiation_result(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result, NegotiationResult)

    def test_result_has_boolean_flags(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result.timedout, bool)
        assert isinstance(result.broken, bool)

    def test_result_steps_is_positive(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert result.steps >= 0

    def test_agreement_covers_all_issues_when_reached(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        if result.agreement is not None:
            assert len(result.agreement) == len(issues)
            agreed_ids = {o.issue_id for o in result.agreement}
            assert agreed_ids == set(issues)

    def test_agreement_options_are_valid(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        if result.agreement is not None:
            for outcome in result.agreement:
                assert outcome.chosen_option in options_per_issue[outcome.issue_id]

    def test_history_is_recorded(
        self, model, issues, options_per_issue, two_participants
    ):
        result = model.run(issues, options_per_issue, two_participants)
        assert isinstance(result.history, list)

    # ------------------------------------------------------------------
    # Identical preferences — agreement should be fast
    # ------------------------------------------------------------------

    def test_identical_preferences_reach_agreement(
        self, issues, options_per_issue
    ):
        prefs = {
            "budget": {"low": 0.1, "medium": 0.9, "high": 0.0},
            "timeline": {"short": 0.0, "long": 1.0},
        }
        participants = [
            NegotiationParticipant(id="a", name="A", preferences=prefs),
            NegotiationParticipant(id="b", name="B", preferences=prefs),
        ]
        result = NegotiationModel(n_steps=30).run(issues, options_per_issue, participants)
        assert result.agreement is not None

    # ------------------------------------------------------------------
    # Issue weights
    # ------------------------------------------------------------------

    def test_custom_issue_weights_accepted(
        self, model, issues, options_per_issue
    ):
        participants = [
            NegotiationParticipant(
                id="a",
                name="A",
                preferences={"budget": {"medium": 1.0}, "timeline": {"short": 1.0}},
                issue_weights={"budget": 0.8, "timeline": 0.2},
            ),
            NegotiationParticipant(
                id="b",
                name="B",
                preferences={"budget": {"medium": 1.0}, "timeline": {"short": 1.0}},
                issue_weights={"budget": 0.5, "timeline": 0.5},
            ),
        ]
        result = model.run(issues, options_per_issue, participants)
        assert isinstance(result, NegotiationResult)
