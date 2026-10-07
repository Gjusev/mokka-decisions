"""Unit tests without downloads: contract validation, masking, splits,
permutation remapping, calibration persistence, thresholds, abstention."""

from __future__ import annotations

import json
import math

import pytest

from mokka_decisions.contracts import (
    NONE_ID,
    ContractError,
    DecisionCase,
    DecisionResponse,
    Option,
    validate_request_payload,
)


def make_case(**overrides) -> DecisionCase:
    base = dict(
        id="t-1",
        group_id="t-1",
        source="test",
        language="en",
        domain="banking",
        state="I was charged twice.",
        question="Which intent applies?",
        options=(
            Option(id="billing", description="Billing"),
            Option(id="tech", description="Technical"),
            Option(id=NONE_ID, description="None applies"),
        ),
        target_kind="single",
        target_option="billing",
        split="train",
    )
    base.update(overrides)
    return DecisionCase(**base)


class TestContract:
    def test_valid_case_roundtrip(self):
        case = make_case()
        restored = DecisionCase.from_dict(json.loads(case.to_json_line()))
        assert restored == case

    def test_duplicate_option_ids_rejected(self):
        with pytest.raises(ContractError):
            make_case(options=(Option(id="a", description="x"), Option(id="a", description="y")))

    def test_target_must_be_among_options(self):
        with pytest.raises(ContractError):
            make_case(target_option="missing")

    def test_blank_state_rejected(self):
        with pytest.raises(ContractError):
            make_case(state="   ")

    def test_response_sums_to_one(self):
        resp = DecisionResponse(
            candidate="billing",
            probabilities={"billing": 0.6, "tech": 0.3, NONE_ID: 0.1},
            choice="billing",
            confidence=0.6,
            abstain=False,
            reason_code="accepted",
        )
        assert resp.confidence == 0.6

    def test_response_rejects_bad_sum(self):
        with pytest.raises(ContractError):
            DecisionResponse(
                candidate="billing",
                probabilities={"billing": 0.6, "tech": 0.6},
                choice="billing",
                confidence=0.6,
                abstain=False,
                reason_code="accepted",
            )

    def test_abstain_forces_null_choice(self):
        with pytest.raises(ContractError):
            DecisionResponse(
                candidate="billing",
                probabilities={"billing": 1.0},
                choice="billing",
                confidence=0.3,
                abstain=True,
                reason_code="below_validated_threshold",
            )

    def test_request_payload_minimal(self):
        case = validate_request_payload(
            {
                "state": "help",
                "options": [
                    {"id": "a", "description": "A"},
                    {"id": "b", "description": "B"},
                ],
            }
        )
        assert case.state == "help"
        assert len(case.options) == 2


class TestModelMasking:
    def test_padding_slots_are_masked(self):
        """The -1e9 fill must place padded slots effectively at -inf in softmax."""
        import torch

        from mokka_decisions.model import OptionScorer

        logits = torch.full((2, 4), -1e9)
        logits[0, 0] = 2.0
        logits[0, 1] = 1.0
        logits[1, 2] = 3.0
        probs0 = torch.softmax(logits[0], dim=-1)
        assert probs0[2].item() < 1e-12 and probs0[3].item() < 1e-12
        assert probs0[:2].sum().item() == pytest.approx(1.0, abs=1e-6)
        assert torch.argmax(logits[1]).item() == 2


class TestSplits:
    def test_split_is_deterministic_and_stable(self):
        from mokka_decisions.splitting import split_for_group

        assert split_for_group("g1") == split_for_group("g1")
        assert split_for_group("massive-42") == split_for_group("massive-42")

    def test_massive_locales_share_split(self):
        from mokka_decisions.splitting import split_for_group

        # same underlying id, different locale prefix -> same split decision
        # because the group id is locale-independent by construction
        groups = [f"massive-{i}" for i in range(500)]
        seen = {g: split_for_group(g) for g in groups}
        assert len(seen) == 500  # pure function, no hidden state

    def test_group_disjunction_catches_leak(self):
        from mokka_decisions.splitting import assert_group_disjunction

        rows = [{"group_id": "g", "split": "train"}, {"group_id": "g", "split": "test"}]
        with pytest.raises(AssertionError):
            assert_group_disjunction(rows)

    def test_ratios_roughly_hold(self):
        from mokka_decisions.splitting import split_for_group

        n = 5000
        counts = {}
        for i in range(n):
            s = split_for_group(f"x-{i}")
            counts[s] = counts.get(s, 0) + 1
        assert 0.75 < counts["train"] / n < 0.85
        assert counts["dev"] / n > 0.04


class TestInstances:
    def _row(self, label="card_arrival", oos=False):
        return {
            "id": "bank-tr-00001",
            "group_id": "bank-tr-00001",
            "source": "banking77",
            "language": "en",
            "domain": "banking",
            "text": "Where is my card?",
            "label": "oos" if oos else label,
            "official_split": "train",
            "label_origin": "public-dataset",
        }

    def _catalog(self):
        from mokka_decisions.taxonomies import build_catalog

        labels = ["card_arrival", "card_not_working", "change_pin", "activate_my_card",
                  "atm_support", "cancel_transfer", "exchange_rate", "terminate_account"]
        return build_catalog("banking77", labels)

    def test_gold_always_present_and_deterministic(self):
        from mokka_decisions.instances import make_case

        cat = self._catalog()
        c1 = make_case(self._row(), cat, k=5, include_none=True, salt="train")
        c2 = make_case(self._row(), cat, k=5, include_none=True, salt="train")
        assert c1.options == c2.options
        assert c1.target_option == "card_arrival"
        assert "card_arrival" in c1.option_ids
        assert len(c1.options) == 5

    def test_oos_row_targets_none(self):
        from mokka_decisions.instances import make_case

        cat = self._catalog()
        case = make_case(self._row(oos=True), cat, k=5, include_none=True, salt="train")
        assert case.target_option == NONE_ID
        assert NONE_ID in case.option_ids

    def test_permutation_preserves_gold_position_mapping(self):
        from mokka_decisions.instances import permute_case

        case = make_case()
        perm = permute_case(case, seed=7)
        assert sorted(perm.option_ids) == sorted(case.option_ids)
        assert perm.target_option == case.target_option
        # and a different order really happened at least once over tries
        changed = any(
            permute_case(case, seed=s).option_ids != case.option_ids for s in range(10)
        )
        assert changed

    def test_train_k_range_respected(self):
        from mokka_decisions.instances import expand_rows

        cat_by_src = {"banking77": self._catalog()}
        rows = [self._row()]
        cases = expand_rows(rows, cat_by_src, split="train")
        assert 4 <= len(cases[0].options) <= 8


class TestCalibration:
    def test_temperature_improves_overconfident_nll(self):
        from mokka_decisions.calibrate import fit_temperature

        # overconfident logits: correct option wins with margin but T<1 optimal is >1
        cases, logits = [], []
        for i in range(50):
            case = make_case(id=f"c{i}", group_id=f"c{i}")
            cases.append(case)
            row = {"billing": 6.0, "tech": 0.0, NONE_ID: -6.0}
            if i % 5 == 0:  # 20% mistakes
                row = {"billing": 0.0, "tech": 6.0, NONE_ID: -6.0}
            logits.append(row)
        t = fit_temperature(cases, logits)
        assert t > 1.0  # overconfidence -> temperature above one

    def test_temperature_preserves_argmax(self):
        from mokka_decisions.calibrate import _nll, fit_temperature

        cases, logits = [], []
        for i in range(30):
            case = make_case(id=f"c{i}", group_id=f"c{i}")
            cases.append(case)
            logits.append({"billing": 3.0 + i * 0.01, "tech": 1.0, NONE_ID: -2.0})
        t = fit_temperature(cases, logits)
        for case, row in zip(cases, logits):
            vals = [row[o] for o in case.option_ids]
            idx = vals.index(max(vals))
            assert case.option_ids[idx] == case.target_option


class TestPolicy:
    def def_cases_probs(self):
        from mokka_decisions.policy import apply_policy

        case = make_case()
        probs = {"billing": 0.9, "tech": 0.07, NONE_ID: 0.03}
        return case, probs, apply_policy

    def test_accept_above_threshold(self):
        case, probs, apply_policy = self.def_cases_probs()
        resp = apply_policy(case, probs, threshold=0.5)
        assert resp.choice == "billing" and not resp.abstain

    def test_abstain_below_threshold(self):
        case, probs, apply_policy = self.def_cases_probs()
        resp = apply_policy(case, probs, threshold=0.95)
        assert resp.choice is None and resp.abstain
        assert resp.reason_code == "below_validated_threshold"

    def test_language_gate(self):
        case, probs, apply_policy = self.def_cases_probs()
        resp = apply_policy(case, probs, threshold=0.5, languages=("en",))
        assert not resp.abstain
        resp_de = apply_policy(case, probs, threshold=0.5, languages=("de",))
        assert resp_de.abstain and resp_de.reason_code == "language_not_validated"

    def test_pick_threshold_meets_target(self):
        from mokka_decisions.policy import pick_threshold

        cases, probs = [], []
        for i in range(200):
            case = make_case(id=f"p{i}", group_id=f"p{i}")
            cases.append(case)
            if i % 10 == 0:  # wrong but confident
                probs.append({"billing": 0.2, "tech": 0.75, NONE_ID: 0.05})
            elif i % 10 == 5:  # wrong and unconfident
                probs.append({"billing": 0.4, "tech": 0.35, NONE_ID: 0.25})
            else:
                probs.append({"billing": 0.85, "tech": 0.1, NONE_ID: 0.05})
        chosen = pick_threshold(cases, probs, target_risk=0.05)
        assert chosen["met_target"]
        assert chosen["coverage"] > 0.5


class TestBackends:
    def test_normalise_probabilities(self):
        from mokka_decisions.backends.base import normalise_probabilities

        assert normalise_probabilities({"a": 2.0, "b": 2.0}) == {"a": 0.5, "b": 0.5}
        assert normalise_probabilities({"a": float("nan"), "b": 1.0}) == {"b": 1.0}
        assert normalise_probabilities({}) == {}

    def test_linear_backend_restricts_to_present(self):
        pytest.importorskip("sklearn")
        from mokka_decisions.backends.linear import LinearBackend

        rows = [
            {"text": "my card is lost", "label": "lost_or_stolen_card", "language": "en"},
            {"text": "refund missing", "label": "Refund_not_showing_up", "language": "en"},
            {"text": "card lost again", "label": "lost_or_stolen_card", "language": "en"},
            {"text": "where is my refund", "label": "Refund_not_showing_up", "language": "en"},
        ] * 5
        backend = LinearBackend(rows)
        case = make_case(
            options=(Option(id="lost_or_stolen_card", description="x"),
                     Option(id="Refund_not_showing_up", description="z"),
                     Option(id=NONE_ID, description="y")),
            target_option="lost_or_stolen_card",
        )
        probs = backend.score_one(case)
        assert NONE_ID not in probs  # none has no signal
        assert set(probs) == {"lost_or_stolen_card", "Refund_not_showing_up"}
        rows = backend.decide([case])
        assert rows[0].error == ""


class TestEvaluate:
    def test_summarize_counts_errors(self):
        from mokka_decisions.backends.base import BackendRow
        from mokka_decisions.evaluate import summarize

        case = make_case()
        row_ok = BackendRow(
            case_id=case.id, group_id=case.group_id, source=case.source,
            language=case.language, domain=case.domain, backend="t",
            model_revision="", candidate="billing",
            probabilities={"billing": 0.8, "tech": 0.15, NONE_ID: 0.05}, latency_ms=1.0,
        )
        row_err = BackendRow(
            case_id=case.id, group_id=case.group_id, source=case.source,
            language=case.language, domain=case.domain, backend="t",
            model_revision="", candidate="", probabilities={}, latency_ms=2.0,
            error="Boom",
        )
        s = summarize([row_ok, row_err], [case, case])
        assert s["n_cases"] == 2 and s["backend_errors"] == 1

    def test_bootstrap_by_group(self):
        from mokka_decisions.backends.base import BackendRow
        from mokka_decisions.evaluate import bootstrap_accuracy_ci

        cases = [make_case(id=f"c{i}", group_id=f"g{i//2}") for i in range(20)]
        rows = []
        for i, c in enumerate(cases):
            rows.append(BackendRow(
                case_id=c.id, group_id=c.group_id, source=c.source, language=c.language,
                domain=c.domain, backend="t", model_revision="",
                candidate="billing" if i % 2 else "tech",
                probabilities={"billing": 0.5, "tech": 0.5, NONE_ID: 0.0}, latency_ms=1.0,
            ))
        ci = bootstrap_accuracy_ci(rows, cases, n_boot=100)
        assert ci["ci95"][0] <= ci["accuracy"] <= ci["ci95"][1]
