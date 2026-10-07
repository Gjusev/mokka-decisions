"""Synthetic multimodal decision data: Mokka Demo Bank episodes.

An *episode* is (template, parameters) — the app/episode/screen granularity
the split respects BEFORE any variant (language, degradation) is produced.

Case types (the evidence design, each stated in the row):

- ``text_suffices``   the message alone identifies the intent; the screenshot
                      is a generic settings screen or absent;
- ``vision_needed``   the message is vague; the decisive fact (status banner,
                      duplicated rows, rate) exists only in the screenshot;
- ``missing_image``   the decisive fact is visual but no screenshot was
                      attached -> target_kind ``ambiguous`` (no forced-gold;
                      used to measure abstention / ask-for-info behaviour);
- ``blurry``          vision_needed with a degraded screenshot (mild/strong
                      blur or low contrast) — partial evidence;
- ``contradiction``   message and screenshot disagree -> ``ambiguous``;
- ``no_valid_option`` a real issue outside the option set -> gold ``none``.

Labels are authored-by-construction: the generator writes the intent it
encoded; no model or human inferred anything (``label_origin`` states this).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import NONE_ID, DecisionCase
from ..taxonomies import BANKING77_DESCRIPTIONS, build_catalog
from .render import apply_degradation, image_sha256, render_screen, save_png

QUESTION = "Which banking support intent does this customer request express?"

# episode templates -> intent + screen builder params (authored mapping)
TEMPLATES: dict[str, dict[str, Any]] = {
    "card_late": {
        "intent": "card_arrival",
        "screen": ("card_status", "late"),
        "decisive_text": "ordered a new card {days} days ago and it has not arrived",
    },
    "card_lost": {
        "intent": "lost_or_stolen_card",
        "screen": ("card_status", "blocked"),
        "decisive_text": "phone was stolen and the card with it, need it blocked",
    },
    "double_charge": {
        "intent": "transaction_charged_twice",
        "screen": ("tx_history", "duplicate"),
        "decisive_text": "charged twice for the same purchase this morning",
    },
    "pending_transfer": {
        "intent": "pending_transfer",
        "screen": ("tx_history", "pending"),
        "decisive_text": "transfer still shows as pending after days",
    },
    "balance_stale": {
        "intent": "balance_not_updated_after_bank_transfer",
        "screen": ("tx_history", "stale"),
        "decisive_text": "sent a bank transfer days ago and the balance did not change",
    },
    "wrong_rate": {
        "intent": "card_payment_wrong_exchange_rate",
        "screen": ("receipt", "rate"),
        "decisive_text": "paid abroad and the exchange rate applied looks wrong",
    },
    "refund_missing": {
        "intent": "Refund_not_showing_up",
        "screen": ("receipt", "refund"),
        "decisive_text": "refund was promised over a week ago and never arrived",
    },
    "pin_locked": {
        "intent": "pin_blocked",
        "screen": ("card_status", "pin"),
        "decisive_text": "entered the PIN wrong and now the card is locked",
    },
    "topup_failed": {
        "intent": "top_up_failed",
        "screen": ("receipt", "topup"),
        "decisive_text": "tried to top up and it failed but the money left my bank",
    },
    # OOD episodes (outside the offered option set -> none)
    "otp_fail": {"intent": NONE_ID, "screen": ("otp_prompt", "otp"), "decisive_text": "cannot log in, the code does not work"},
    "loan_offer": {"intent": NONE_ID, "screen": ("marketing", "loan"), "decisive_text": "want to know more about the loan offer shown in the app"},
}

CASE_TYPES = (
    "text_suffices",
    "vision_needed",
    "missing_image",
    "blurry",
    "contradiction",
    "no_valid_option",
)

NAMES = ["Marta", "Lukas", "Elena", "Jonas", "Sofia", "Timo", "Nadia", "Paul"]
AMOUNTS = ["4.20", "12.90", "249.00", "1,150.00", "63.50", "38.00"]
CASE_REFS = lambda rng: f"CASE-{rng.randrange(10000, 99999)}"

LANGS = {
    "en": {
        "vague": "Something is wrong with the app, please help.",
        "contradiction": "this was already solved, why does the app still show a problem?",
        "screens": "screens",
    },
    "de": {
        "vague": "Mit der App stimmt etwas nicht, bitte helfen.",
        "contradiction": "das Thema ist doch bereits geklärt, warum zeigt die App immer noch ein Problem an?",
    },
    "es": {
        "vague": "Algo va mal con la app, por favor ayudadme.",
        "contradiction": "este asunto ya estaba resuelto, ¿por qué la app sigue mostrando un problema?",
    },
}


def _screen_params(template: str, variant: str, rng: random.Random, lang: str) -> dict:
    """Authored screen content per template variant; decisive text only here."""
    name = rng.choice(NAMES)
    amount = rng.choice(AMOUNTS)
    ref = CASE_REFS(rng)
    days = rng.choice([9, 11, 12, 14])
    if template == "card_status" and variant == "late":
        return {
            "title": "Cards", "headline": "Your new card",
            "card_name": f"Virtual card • {rng.choice(['4029', '8831', '1174'])}",
            "status_line": "Ordered — not shipped yet", "status_color": (191, 131, 12),
            "detail": "Standard delivery: 5-9 working days",
            "ordered": f"Ordered: {days} days ago",
            "banner": "Delivery is taking longer than usual. We are sorry.",
            "banner_color": (196, 43, 43), "ref": ref,
        }
    if template == "card_status" and variant == "blocked":
        return {
            "title": "Cards", "headline": "Card status",
            "card_name": f"Debit card • {rng.choice(['5560', '2291'])}",
            "status_line": "Permanently blocked", "status_color": (196, 43, 43),
            "detail": "Card reported lost via phone support",
            "ordered": f"Blocked: today, {rng.choice(['09:14', '11:32', '17:45'])}",
            "banner": "A replacement card must be ordered.", "banner_color": (191, 131, 12),
            "ref": ref,
        }
    if template == "card_status" and variant == "pin":
        return {
            "title": "Cards", "headline": "Card security",
            "card_name": "Debit card • 7702",
            "status_line": "PIN locked after 3 wrong attempts", "status_color": (196, 43, 43),
            "detail": "Use the app to unlock or visit an ATM",
            "ordered": f"Locked: today, {rng.choice(['08:02', '13:26'])}",
            "banner": "Contact support to reset your PIN immediately.",
            "banner_color": (191, 131, 12), "ref": ref,
        }
    if template == "tx_history" and variant == "duplicate":
        when = rng.choice(["today, 08:12", "today, 19:04", "yesterday, 12:31"])
        row = (rng.choice(["Coffee Bar", "Bike Store", "Market Hall"]), f"-{amount} EUR", when)
        return {
            "title": "History", "headline": "Transactions",
            "banner": "Duplicate charge detected on your card.",
            "banner_color": (196, 43, 43),
            "rows": [(*row[:2], row[2], (24, 26, 31)),
                     (row[0], row[1], row[2], (24, 26, 31)),
                     ("Salary", "+2,150.00 EUR", "01 Oct", (22, 125, 72))],
            "footnote": f"Reference {ref}",
        }
    if template == "tx_history" and variant == "pending":
        return {
            "title": "History", "headline": "Transactions",
            "banner": f"Transfer of {amount} EUR pending for {rng.choice([3, 4, 6])} days.",
            "banner_color": (191, 131, 12),
            "rows": [("Transfer to Marta L.", f"-{amount} EUR", "pending", (191, 131, 12)),
                     ("Rent", "-840.00 EUR", "28 Sep", (24, 26, 31))],
            "footnote": f"Reference {ref}",
        }
    if template == "tx_history" and variant == "stale":
        return {
            "title": "History", "headline": "Transactions",
            "banner": f"Bank transfer of {amount} EUR sent {rng.choice([3, 5])} days ago — balance unchanged.",
            "banner_color": (196, 43, 43),
            "rows": [("Bank transfer", f"-{amount} EUR", "sent, not reflected", (196, 43, 43)),
                     ("Card payment", f"-{amount} EUR", "30 Sep", (24, 26, 31))],
            "footnote": f"Reference {ref}",
        }
    if template == "receipt" and variant == "rate":
        return {
            "title": "Payment", "headline": "Card payment",
            "fields": [("Merchant", rng.choice(["Blue Bike Store", "Nord Hotel", "Alpine Ski Rent"])),
                       ("Amount", f"{amount} EUR"),
                       ("Rate applied", f"{rng.choice(['1.0871', '1.1930'])} EUR per USD"),
                       ("You were charged", f"{amount} EUR")],
            "banner": "Rate differs from the one shown at checkout.",
            "banner_color": (196, 43, 43), "ref": f"AUTH-{rng.randrange(10000, 99999)}",
        }
    if template == "receipt" and variant == "refund":
        return {
            "title": "Refund", "headline": "Refund status",
            "fields": [("Merchant", rng.choice(["Blue Bike Store", "Nord Hotel"])),
                       ("Refund amount", f"{amount} EUR"),
                       ("Issued", f"{rng.choice([8, 10, 12])} days ago"),
                       ("Status", "Not received")],
            "banner": "The refund has not reached your account.",
            "banner_color": (196, 43, 43), "ref": CASE_REFS(rng),
        }
    if template == "receipt" and variant == "topup":
        return {
            "title": "Top-up", "headline": "Top-up attempt",
            "fields": [("Method", "Bank transfer"),
                       ("Amount", f"{amount} EUR"),
                       ("Debited from bank", "Yes"),
                       ("Status", "Failed")],
            "banner": "Top-up failed after the money left your bank.",
            "banner_color": (196, 43, 43), "ref": CASE_REFS(rng),
        }
    if template == "otp_prompt":
        return {
            "title": "Login", "headline": "Verification required",
            "body": "Enter the 6-digit code we sent to your phone.",
            "hint": "The code expires after 5 minutes.",
            "banner": f"Code not valid. Attempt {rng.choice([2, 3])} of 3.",
            "banner_color": (196, 43, 43),
        }
    if template == "marketing":
        return {
            "title": "Offers", "headline": "For you",
            "offer": f"Personal loan up to {rng.choice(['10,000', '15,000'])} EUR",
            "detail": "Instant decision, no paperwork.",
            "apr": "6.9% APR representative", "cta": "T&Cs apply. 18+.",
        }
    raise KeyError(f"unknown screen variant {template}/{variant}")


def _message_for(kind: str, spec: dict, rng: random.Random, lang: str) -> str:
    vague = LANGS[lang]["vague"]
    contradiction = LANGS[lang]["contradiction"]
    decisive = spec["decisive_text"]
    if kind == "text_suffices":
        return f"I {decisive}."
    if kind in ("vision_needed", "blurry"):
        return vague
    if kind == "missing_image":
        return vague
    if kind == "contradiction":
        return contradiction
    if kind == "no_valid_option":
        return f"I {decisive}."
    raise KeyError(kind)


def _stable_seed(*parts) -> int:
    """Deterministic across runs/platforms (Python's hash() is salted)."""
    key = "|".join(str(p_) for p_ in parts)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


@dataclass
class Episode:
    template_key: str
    seed: int
    case_type: str
    group_id: str


def build_dataset(
    *,
    out_dir: str | Path,
    n_episodes: int = 360,
    train_ratio: float = 0.68,
    dev_ratio: float = 0.08,
    cal_temperature_ratio: float = 0.08,
    cal_policy_ratio: float = 0.08,
    seed: int = 42,
    langs: tuple[str, ...] = ("en", "de", "es"),
    k_options: int = 5,
) -> dict:
    """Build the synthetic multimodal dataset with episode-level splits.

    Splits happen on episode (template+params) BEFORE language/degradation
    variants are produced, using the canonical split names (train / dev /
    cal_temperature / cal_policy / test). Images land in images/<split>/;
    cases in cases/<split>.jsonl with media hashes. Deterministic per seed.
    """
    out = Path(out_dir)
    rng = random.Random(seed)
    catalog = build_catalog("banking77", sorted(BANKING77_DESCRIPTIONS))

    # episode plan: cycle templates; case types drawn per-episode (weighted)
    type_cycle = ["text_suffices", "vision_needed", "vision_needed", "missing_image",
                  "blurry", "contradiction", "no_valid_option"]
    template_keys = sorted(TEMPLATES)
    episodes = []
    for i in range(n_episodes):
        key = template_keys[i % len(template_keys)]
        # OOD templates only make sense as no_valid_option (or contradiction)
        kind = type_cycle[i % len(type_cycle)]
        if TEMPLATES[key]["intent"] == NONE_ID and kind not in ("no_valid_option", "text_suffices"):
            kind = "no_valid_option"
        if TEMPLATES[key]["intent"] != NONE_ID and kind == "no_valid_option":
            kind = "text_suffices"
        episodes.append(
            Episode(
                template_key=key,
                seed=seed * 7919 + i * 104729,
                case_type=kind,
                group_id=f"mm-{key}-{i:04d}",
            )
        )

    # split episodes (not variants) by hash
    def split_of(group_id: str) -> str:
        u = int.from_bytes(hashlib.sha256(f"mm-split-v1|{group_id}".encode()).digest()[:8], "big") / 2**64
        if u < train_ratio:
            return "train"
        if u < train_ratio + dev_ratio:
            return "dev"
        if u < train_ratio + dev_ratio + cal_temperature_ratio:
            return "cal_temperature"
        if u < train_ratio + dev_ratio + cal_temperature_ratio + cal_policy_ratio:
            return "cal_policy"
        return "test"

    counts = {"train": 0, "dev": 0, "cal_temperature": 0, "cal_policy": 0, "test": 0}
    case_files = {s: [] for s in counts}
    image_records = []
    for ep in episodes:
        spec = TEMPLATES[ep.template_key]
        screen_template, screen_variant = spec["screen"]
        split = split_of(ep.group_id)
        intent = spec["intent"]

        for lang in langs:
            vrng = random.Random(_stable_seed(ep.seed, lang, ep.case_type))
            message = _message_for(ep.case_type, spec, vrng, lang)

            img_rel = None
            img = None
            if ep.case_type in ("text_suffices", "no_valid_option"):
                # irrelevant/absent image; ~50% of text_suffices get a settings screen
                if vrng.random() < 0.5:
                    img = render_screen("settings", {"title": "Settings"})
            elif ep.case_type == "vision_needed":
                img = render_screen(screen_template, _screen_params(screen_template, screen_variant, vrng, lang))
            elif ep.case_type == "blurry":
                base = render_screen(screen_template, _screen_params(screen_template, screen_variant, vrng, lang))
                img = apply_degradation(base, vrng.choice(["blur_mild", "blur_strong", "lowcontrast"]), ep.seed)
            elif ep.case_type == "missing_image":
                img = None  # decisive info exists only in the un-attached screen
            elif ep.case_type == "contradiction":
                # message claims all-fine; screen shows the problem
                img = render_screen(screen_template, _screen_params(screen_template, screen_variant, vrng, lang))

            if img is not None:
                img_path = out / "images" / split / f"{ep.group_id}-{lang}.png"
                save_png(img, img_path)
                sha = image_sha256(img)
                img_rel = str(img_path.relative_to(out))
                image_records.append({"path": img_rel, "sha256": sha, "episode": ep.group_id, "lang": lang})

            # options: gold + distractors + none (intent OOD -> gold none)
            gold = NONE_ID if ep.case_type == "no_valid_option" else intent
            target_kind = "ambiguous" if ep.case_type in ("missing_image", "contradiction") else ("none" if gold == NONE_ID else "single")
            pool = [i for i in BANKING77_DESCRIPTIONS if i != intent]
            distractors = rng.sample(pool, k_options - 1)
            ids = distractors + [NONE_ID] + ([] if gold == NONE_ID else [intent])
            rng.shuffle(ids)

            case = DecisionCase(
                id=f"{ep.group_id}-{lang}-{ep.case_type}",
                group_id=ep.group_id,
                source="mokka-demo-bank-v1",
                language=lang,
                domain="banking",
                state=message,
                question=QUESTION,
                options=tuple(catalog[oid] for oid in ids),
                target_kind=target_kind,
                target_option=None if target_kind == "ambiguous" else gold,
                label_origin="synthetic-authored-programmatic",
                split=split,
            )
            row = case.to_dict()
            row["case_type"] = ep.case_type
            row["template"] = ep.template_key
            row["media"] = {"screenshot": {"path": img_rel, "sha256": None if img is None else sha}} if (img is not None or ep.case_type == "missing_image") else None
            case_files[split].append(row)
            counts[split] += 1

    for split, rows in case_files.items():
        p = out / "cases" / f"{split}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "seed": seed,
        "n_episodes": n_episodes,
        "counts_by_split": counts,
        "case_types": {t: sum(1 for rows in case_files.values() for r in rows if r["case_type"] == t) for t in CASE_TYPES},
        "images": len(image_records),
        "splits": "episodes split by hash BEFORE language/degradation variants",
        "labels": "authored-by-construction (generator encodes the intent); synthetic, not human-reviewed",
        "templates": {k: v["intent"] for k, v in TEMPLATES.items()},
        "licenses": "synthetic, this project (Apache-2.0 with the code)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "images.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for rec in image_records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    return manifest
