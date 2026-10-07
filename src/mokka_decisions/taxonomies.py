"""Option catalogs: canonical intent ids with authored, readable descriptions.

Descriptions are authored for this project (English, one line) and keyed by the
*exact* label spellings of the source datasets. ``build_catalog`` fails loudly
when a dataset label has no authored description, so a renamed upstream label
breaks the build instead of silently producing an undescribed option.

CLINC150 uses humanized intent names (``transfer_money`` -> ``Transfer money``)
instead of authored glosses; it serves as the out-of-scope and unseen-domain
probe, not as a headline domain. Documented in docs/dataset-card.md.
"""

from __future__ import annotations

from typing import Mapping

from .contracts import NONE_DESCRIPTION, NONE_ID, Option

DOMAIN_QUESTION = {
    "banking77": "Which banking support intent does this customer request express?",
    "massive": "Which task or intent does this utterance express?",
    "clinc": "Which task or intent does this request express?",
}

# ---------------------------------------------------------------------------
# BANKING77 (77 intents, exact HuggingFace PolyAI/banking77 spellings)
# ---------------------------------------------------------------------------
BANKING77_DESCRIPTIONS: dict[str, str] = {
    "activate_my_card": "Activate a card that has just been received",
    "age_limit": "Ask about the minimum age to open an account or get a card",
    "apple_pay_or_google_pay": "Add a card to Apple Pay or Google Pay, or issues paying with them",
    "atm_support": "Find an ATM or get help using an ATM",
    "automatic_top_up": "Set up or change automatic top-ups for the account",
    "balance_not_updated_after_bank_transfer": "Balance did not update after a bank transfer was sent",
    "balance_not_updated_after_cheque_or_cash_deposit": "Balance did not update after depositing a cheque or cash",
    "beneficiary_not_allowed": "A beneficiary or payee cannot be added or is not allowed",
    "cancel_transfer": "Cancel or reverse a transfer that was already sent",
    "card_about_to_expire": "Ask about a card that is about to expire or renewing it",
    "card_acceptance": "Ask where a card is accepted as payment",
    "card_arrival": "Ask when a new or replacement card will arrive",
    "card_delivery_estimate": "Ask for the estimated delivery date of a card",
    "card_linking": "Link or unlink a card to another account, wallet or service",
    "card_not_working": "A card does not work when paying or withdrawing",
    "card_payment_fee_charged": "A fee was charged for a card payment",
    "card_payment_not_recognised": "A card payment on the statement is not recognised",
    "card_payment_wrong_exchange_rate": "A card payment used a wrong exchange rate",
    "card_swallowed": "A card was swallowed or retained by an ATM or machine",
    "cash_withdrawal_not_recognised": "A cash withdrawal on the statement is not recognised",
    "cash_withdrawal_charge": "A fee was charged for withdrawing cash",
    "change_pin": "Change the PIN of a card",
    "compromised_card": "Suspect a card or account has been compromised or used fraudulently",
    "contactless_not_working": "Contactless payment does not work",
    "country_support": "Ask whether the service works in or supports a given country",
    "disposable_card_limits": "Ask about limits for disposable virtual cards",
    "declined_card_payment": "A card payment was declined",
    "declined_cash_withdrawal": "A cash withdrawal was declined",
    "declined_transfer": "A transfer was declined",
    "direct_debit_payment_not_recognised": "A direct debit on the statement is not recognised",
    "edit_personal_details": "Change name, address, phone or other personal details",
    "exchange_charge": "A fee was charged for a currency exchange",
    "exchange_rate": "Ask about the exchange rate applied to a transaction",
    "exchange_via_app": "Exchange or convert currency inside the app",
    "extra_charge_on_statement": "An unexpected extra charge appears on the statement",
    "failed_transfer": "A transfer failed to go through",
    "fiat_currency_support": "Ask which fiat currencies the account supports",
    "get_disposable_virtual_card": "Get a disposable or one-time virtual card",
    "get_physical_card": "Get or request a physical card",
    "getting_spare_card": "Get an additional or spare card for the account",
    "getting_virtual_card": "Get or set up a virtual card",
    "lost_or_stolen_card": "Report a card as lost or stolen, or block it",
    "lost_or_stolen_phone": "Report a lost or stolen phone that had the account, or secure the account",
    "order_physical_card": "Order a physical card",
    "passcode_forgotten": "Forgot the app passcode or cannot log in",
    "pending_card_payment": "A card payment is stuck as pending",
    "pending_cash_withdrawal": "A cash withdrawal is stuck as pending",
    "pending_top_up": "A top-up is stuck as pending",
    "pending_transfer": "A transfer is stuck as pending",
    "pin_blocked": "The card PIN is blocked after wrong attempts",
    "receiving_money": "Ask about receiving money or why incoming money has not arrived",
    "Refund_not_showing_up": "A refund that was promised has not appeared in the account",
    "request_refund": "Request a refund for a purchase or charge",
    "reverted_card_payment?": "Ask why a card payment was reverted or returned",
    "supported_cards_and_currencies": "Ask which cards or currencies are supported",
    "terminate_account": "Close or terminate the account",
    "top_up_by_bank_transfer_charge": "A fee was charged for topping up by bank transfer",
    "top_up_by_card_charge": "A fee was charged for topping up by card",
    "top_up_by_cash_or_cheque": "Top up the account using cash or a cheque",
    "top_up_failed": "A top-up failed to complete",
    "top_up_limits": "Ask about limits on how much can be topped up",
    "top_up_reverted": "A top-up was reversed or reverted",
    "topping_up_by_card": "Top up the account using a card",
    "transaction_charged_twice": "The same transaction was charged twice",
    "transfer_fee_charged": "A fee was charged for a transfer",
    "transfer_into_account": "Money transferred into the account has not arrived",
    "transfer_not_received_by_recipient": "A recipient did not receive a transfer that was sent",
    "transfer_timing": "Ask how long a transfer takes to arrive",
    "unable_to_verify_identity": "Identity verification failed or cannot be completed",
    "verify_my_identity": "Verify identity for the account or a payment",
    "verify_source_of_funds": "Verify or justify the source of funds",
    "verify_top_up": "Verify or confirm a top-up, for example with a code",
    "virtual_card_not_working": "A virtual card does not work when paying online",
    "visa_or_mastercard": "Ask whether Visa or Mastercard is supported",
    "why_verify_identity": "Ask why identity verification is required",
    "wrong_amount_of_cash_received": "Received the wrong amount of cash from an ATM or withdrawal",
    "wrong_exchange_rate_for_cash_withdrawal": "A cash withdrawal used a wrong exchange rate",
}

# The two canonical quirks also exist in lowercase/other spellings in some
# exports; normalise nothing — key by what the dataset actually contains.
BANKING77_ALIASES: dict[str, str] = {
    "refund_not_showing_up": "Refund_not_showing_up",
    "reverted_card_payment": "reverted_card_payment?",
    "card_arrival": "card_arrival",
}

# ---------------------------------------------------------------------------
# MASSIVE (60 intents, exact AmazonScience/massive spellings)
# ---------------------------------------------------------------------------
MASSIVE_DESCRIPTIONS: dict[str, str] = {
    "alarm_query": "Ask what alarms are set or when an alarm rings",
    "alarm_remove": "Remove or delete an alarm",
    "alarm_set": "Set, change or cancel an alarm",
    "audio_volume_down": "Turn the audio volume down",
    "audio_volume_mute": "Mute or unmute the audio",
    "audio_volume_other": "Other request about audio volume levels",
    "audio_volume_up": "Turn the audio volume up",
    "calendar_query": "Ask what is on the calendar or check a scheduled event",
    "calendar_remove": "Remove an event from the calendar",
    "calendar_set": "Create or schedule a calendar event",
    "cooking_query": "Ask about cooking times, substitutions or how to cook something",
    "cooking_recipe": "Ask for a recipe for a dish",
    "datetime_convert": "Convert a date or time between zones or formats",
    "datetime_query": "Ask for the current date, time or day",
    "email_addcontact": "Add a contact to the address book",
    "email_query": "Search or ask about emails received or sent",
    "email_querycontact": "Look up a contact in the address book",
    "email_sendemail": "Compose and send an email",
    "general_greet": "Greet the assistant, for example hello or good morning",
    "general_joke": "Ask to hear a joke",
    "general_quirky": "Chitchat or a quirky request that fits no other category",
    "iot_cleaning": "Control a connected cleaning device such as a robot vacuum",
    "iot_coffee": "Control a connected coffee machine",
    "iot_hue_lightchange": "Change the colour or scene of connected lights",
    "iot_hue_lightdim": "Dim or brighten connected lights",
    "iot_hue_lightoff": "Turn connected lights off",
    "iot_hue_lighton": "Turn connected lights on",
    "iot_hue_lightup": "Increase the brightness of connected lights",
    "iot_wemo_off": "Turn a connected WeMo device off",
    "iot_wemo_on": "Turn a connected WeMo device on",
    "lists_createoradd": "Create a list or add an item to a list",
    "lists_query": "Ask what is on a list",
    "lists_remove": "Remove an item from a list",
    "music_dislikeness": "Give negative feedback on the music playing",
    "music_likeness": "Like or give positive feedback on the music playing",
    "music_query": "Ask about a song, artist or playlist",
    "music_settings": "Change music or playback settings",
    "news_query": "Ask for news about a topic",
    "play_audiobook": "Play an audiobook",
    "play_game": "Play a game",
    "play_music": "Play music, a song, album or playlist",
    "play_podcasts": "Play a podcast episode",
    "play_radio": "Play a radio station or channel",
    "qa_currency": "Ask about a currency exchange rate or conversion",
    "qa_definition": "Ask for the definition or meaning of a word or concept",
    "qa_factoid": "Ask a factual question with a short answer",
    "qa_maths": "Ask a maths or arithmetic question",
    "qa_stock": "Ask about a stock price or market value",
    "recommendation_events": "Ask for event recommendations",
    "recommendation_locations": "Ask for place, restaurant or location recommendations",
    "recommendation_movies": "Ask for movie, show or media recommendations",
    "social_post": "Post or publish a message on social media",
    "social_query": "Ask about posts, messages or activity on social media",
    "takeaway_order": "Order food or a takeaway delivery",
    "takeaway_query": "Ask about takeaway options, status or an order",
    "transport_query": "Ask about a transport service, schedule or connection",
    "transport_taxi": "Order, cancel or manage a taxi or ride",
    "transport_ticket": "Buy or manage transport tickets",
    "transport_traffic": "Ask about traffic conditions or a route",
    "weather_query": "Ask about the weather forecast or conditions",
}

# ---------------------------------------------------------------------------
# CLINC150 — humanized names only (see module docstring)
# ---------------------------------------------------------------------------


def humanize_intent(name: str) -> str:
    """``transfer_money`` -> ``Transfer money``; keeps '?' and alphanumerics."""
    text = name.replace("_", " ").strip()
    return text[0].upper() + text[1:] if text else text


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------


def bank_option(label: str) -> Option:
    key = BANKING77_ALIASES.get(label, label)
    if key in BANKING77_DESCRIPTIONS:
        return Option(id=label, description=BANKING77_DESCRIPTIONS[key])
    raise KeyError(
        f"banking77 label {label!r} has no authored description; "
        f"add it to taxonomies.BANKING77_DESCRIPTIONS"
    )


def massive_option(intent: str) -> Option:
    if intent in MASSIVE_DESCRIPTIONS:
        return Option(id=intent, description=MASSIVE_DESCRIPTIONS[intent])
    raise KeyError(
        f"massive intent {intent!r} has no authored description; "
        f"add it to taxonomies.MASSIVE_DESCRIPTIONS"
    )


def clinc_option(intent: str) -> Option:
    return Option(id=intent, description=humanize_intent(intent))


def none_option() -> Option:
    return Option(id=NONE_ID, description=NONE_DESCRIPTION)


def build_catalog(source: str, labels: list[str]) -> dict[str, Option]:
    """Build id -> Option for a dataset's labels, with the shared none option."""
    builders = {
        "banking77": bank_option,
        "massive": massive_option,
        "clinc": clinc_option,
    }
    builder = builders[source]
    catalog = {label: builder(label) for label in labels}
    catalog[NONE_ID] = none_option()
    return catalog


def catalog_stats(catalog: Mapping[str, Option]) -> str:
    return f"{len(catalog)} options (incl. none)"
