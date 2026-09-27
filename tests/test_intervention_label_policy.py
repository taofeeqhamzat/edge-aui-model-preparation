"""
Unit Tests for Scripted Intervention-Target Label Policy (Phase B, Task 10.3).

Verifies:
1. Determinism: identical inputs produce byte-identical labels and IDs.
2. 5-class intervention vocabulary compliance: strictly ['simplify_options', 'highlight_primary_action',
   'offer_assistance', 'expand_tooltip', 'no_op'].
3. Policy configuration reactivity: altering configuration changes derived labels measurably.
4. Correct majority baseline and class distribution calculation.
5. Strict adherence to behavioral vocabulary (zero affect/confusion labels).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from intervention_label_policy import (
    LABEL_POLICY_VERSION,
    INTERVENTION_VOCABULARY,
    INTERVENTION_TO_ID,
    DEFAULT_OUTCOME_TO_INTERVENTION,
    derive_intervention_label,
    assign_intervention_label,
    compute_label_distribution,
)


class TestInterventionLabelPolicy:
    def test_vocabulary_contract(self):
        expected_vocab = (
            "simplify_options",
            "highlight_primary_action",
            "offer_assistance",
            "expand_tooltip",
            "no_op",
        )
        assert INTERVENTION_VOCABULARY == expected_vocab
        assert len(INTERVENTION_TO_ID) == 5
        for name, idx in INTERVENTION_TO_ID.items():
            assert INTERVENTION_VOCABULARY[idx] == name

    def test_deterministic_mapping(self):
        """The policy must be 100% deterministic and pure."""
        context = {
            "route": "Analytics",
            "primaryActionAvailable": False,
            "helpAvailable": True,
            "expandable": True,
            "taskId": "T1",
            "taskStepId": "T1-2",
        }

        res1 = assign_intervention_label("HOVER_DWELL", context)
        res2 = assign_intervention_label("HOVER_DWELL", context)

        assert res1 == res2
        assert res1["target_intervention"] == "expand_tooltip"
        assert res1["target_intervention_id"] == 3
        assert res1["label_policy_version"] == LABEL_POLICY_VERSION
        assert res1["is_scripted_policy"] is True

    def test_context_conditioned_highlight_primary_action(self):
        """When user dwells while primary action is available late in task, recommend highlight_primary_action."""
        # Task progress > 0.5 (T1 step 4 of 4: progress = 0.75)
        context_late = {
            "route": "Analytics",
            "primaryActionAvailable": True,
            "taskProgress": 0.75,
            "taskId": "T1",
            "taskStepId": "T1-4",
        }
        label, label_id = derive_intervention_label("HOVER_DWELL", context_late)
        assert label == "highlight_primary_action"
        assert label_id == INTERVENTION_TO_ID["highlight_primary_action"]

        # Early step: progress < 0.5 (T1 step 2 of 4: progress = 0.25)
        context_early = {
            "route": "Analytics",
            "primaryActionAvailable": True,
            "taskProgress": 0.25,
            "taskId": "T1",
            "taskStepId": "T1-2",
        }
        label_early, _ = derive_intervention_label("HOVER_DWELL", context_early)
        assert label_early == "expand_tooltip"

    def test_outcome_fallbacks(self):
        assert derive_intervention_label("BACKTRACK")[0] == "offer_assistance"
        assert derive_intervention_label("ABANDON")[0] == "offer_assistance"
        assert derive_intervention_label("RAPID_SCROLL")[0] == "simplify_options"
        assert derive_intervention_label("CLICK")[0] == "no_op"
        assert derive_intervention_label("FORM_SUBMIT")[0] == "no_op"
        assert derive_intervention_label("NO_OUTCOME")[0] == "no_op"
        assert derive_intervention_label("UNKNOWN_OUTCOME")[0] == "no_op"

    def test_custom_policy_override_reactivity(self):
        """Changing the policy configuration changes the assigned labels measurably."""
        default_label, _ = derive_intervention_label("RAPID_SCROLL")
        assert default_label == "simplify_options"

        # Override policy mapping RAPID_SCROLL -> offer_assistance
        custom_policy = {"RAPID_SCROLL": "offer_assistance"}
        overridden_label, overridden_id = derive_intervention_label(
            "RAPID_SCROLL", custom_policy=custom_policy
        )

        assert overridden_label == "offer_assistance"
        assert overridden_id == INTERVENTION_TO_ID["offer_assistance"]
        assert overridden_label != default_label

    def test_class_distribution_computation(self):
        labels = [
            "no_op",
            "no_op",
            "no_op",
            "expand_tooltip",
            "highlight_primary_action",
        ]
        dist = compute_label_distribution(labels)

        assert dist["total_examples"] == 5
        assert dist["counts"]["no_op"] == 3
        assert dist["counts"]["expand_tooltip"] == 1
        assert dist["counts"]["highlight_primary_action"] == 1
        assert dist["counts"]["simplify_options"] == 0
        assert dist["counts"]["offer_assistance"] == 0

        assert dist["percentages"]["no_op"] == pytest.approx(60.0)
        assert dist["percentages"]["expand_tooltip"] == pytest.approx(20.0)
        assert dist["majority_class"] == "no_op"
        assert dist["majority_percentage"] == pytest.approx(60.0)
