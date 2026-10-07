"""Two-minute demo: three decisions through the trained model, one of them OOS.

Run from the repo root after a release exists::

    python examples/demo.py --model-dir release/mokka-decisions-v0.1.0

Shows the contract end to end: probabilities (calibrated), accepted choice vs
abstention, and reason codes. Falls back to the untrained head (random
encoder) with --random only to smoke-test wiring without a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = [
    {
        "label": "EN · banking, clean case",
        "request": {
            "state": "I bought a coffee this morning and I see two charges for it on my card.",
            "question": "Which banking support intent does this customer request express?",
            "options": [
                {"id": "transaction_charged_twice", "description": "The same transaction was charged twice"},
                {"id": "card_payment_not_recognised", "description": "A card payment on the statement is not recognised"},
                {"id": "top_up_reverted", "description": "A top-up was reversed or reverted"},
                {"id": "exchange_charge", "description": "A fee was charged for a currency exchange"},
                {"id": "none", "description": "None of the listed options applies to this request."},
            ],
        },
    },
    {
        "label": "DE · banking, cross-lingual transfer",
        "request": {
            "state": "Meine Karte ist letzte Woche gestohlen worden, ich brauche sofort eine neue.",
            "question": "Which banking support intent does this customer request express?",
            "options": [
                {"id": "lost_or_stolen_card", "description": "Report a card as lost or stolen, or block it"},
                {"id": "card_arrival", "description": "Ask when a new or replacement card will arrive"},
                {"id": "change_pin", "description": "Change the PIN of a card"},
                {"id": "visa_or_mastercard", "description": "Ask whether Visa or Mastercard is supported"},
                {"id": "none", "description": "None of the listed options applies to this request."},
            ],
        },
    },
    {
        "label": "ES · out of scope (should pick none or abstain)",
        "request": {
            "state": "¿Me puedes recomendar una película para esta noche?",
            "question": "Which banking support intent does this customer request express?",
            "options": [
                {"id": "terminate_account", "description": "Close or terminate the account"},
                {"id": "receiving_money", "description": "Ask about receiving money or why incoming money has not arrived"},
                {"id": "atm_support", "description": "Find an ATM or get help using an ATM"},
                {"id": "none", "description": "None of the listed options applies to this request."},
            ],
        },
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    from mokka_decisions.contracts import validate_request_payload
    from mokka_decisions.export import load_exported_model
    from mokka_decisions.policy import apply_policy

    scorer, policy = load_exported_model(Path(args.model_dir))
    for demo in CASES:
        case = validate_request_payload(demo["request"])
        probs = scorer.score([case])[0]
        response = apply_policy(
            case,
            probs,
            threshold=policy["threshold"],
            model_revision=scorer.model_revision,
            calibration_revision=policy.get("calibration_revision", ""),
        )
        print(f"\n=== {demo['label']} ===")
        print(f"state: {case.state}")
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        for option_id, p in top:
            print(f"  {option_id:<36} {p:.3f}")
        outcome = (
            f"choice={response.choice}"
            if not response.abstain
            else f"ABSTAIN ({response.reason_code})"
        )
        print(f"-> {outcome}  confidence={response.confidence:.3f}")
        print(json.dumps(response.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
