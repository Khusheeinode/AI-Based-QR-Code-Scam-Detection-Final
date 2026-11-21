"""
text_analysis.py

Heuristic risk scoring for payload types that are not URL or UPI:
plain text, vCard, Wi-Fi credential strings.

None of these can be verified automatically, so every result carries
analysis_status="unverified". A low score means "nothing suspicious was
found", not "safe".
"""

import re

SCAM_KEYWORDS = [
    "urgent", "verify", "suspended", "claim now", "limited time", "act now",
    "kyc update", "account blocked", "won a prize", "congratulations",
    "click here", "otp", "refund",
]

# Whole-word / whole-phrase match, so "otp" does not hit "hotpot".
_KEYWORD_PATTERNS = [
    (kw, re.compile(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"))
    for kw in SCAM_KEYWORDS
]

EMBEDDED_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
EMBEDDED_PHONE_PATTERN = re.compile(r"\b\d{10}\b")

UNVERIFIED_NOTE = (
    "This kind of content cannot be verified automatically, so a low "
    "score does not mean it is safe."
)


def _keyword_hits(text):
    return [kw for kw, pattern in _KEYWORD_PATTERNS if pattern.search(text)]


def analyze_text(payload: str) -> dict:
    text = payload.lower()
    reasons = []
    risk = 0.0

    hits = _keyword_hits(text)
    if hits:
        reasons.append(
            f"Contains scam-associated phrasing: {', '.join(hits[:3])}"
        )
        risk += 0.15 * len(hits)

    if EMBEDDED_URL_PATTERN.search(payload):
        reasons.append(
            "Plain-text payload contains an embedded link. "
            "Check the link before opening it."
        )
        risk += 0.2

    if EMBEDDED_PHONE_PATTERN.search(payload) and any(
        word in text for word in ("call", "contact", "helpline")
    ):
        reasons.append(
            "Contains an embedded phone number paired with an urgency "
            "call-to-action"
        )
        risk += 0.15

    reasons.append(UNVERIFIED_NOTE)

    return {
        "risk_score": min(risk, 1.0),
        "reasons": reasons,
        "analysis_status": "unverified",
    }


def analyze_vcard(payload: str) -> dict:
    reasons = []
    risk = 0.0

    if re.search(r"^URL[;:]", payload, re.IGNORECASE | re.MULTILINE):
        if EMBEDDED_URL_PATTERN.search(payload):
            reasons.append(
                "vCard contains an embedded URL field. "
                "Check the link before opening it."
            )
            risk += 0.2

    # Real name fields only: "FN:" or "N:" at the start of a line
    # (the old check also matched the "N:" inside "BEGIN:VCARD").
    if not re.search(r"^(FN|N)[;:]", payload, re.IGNORECASE | re.MULTILINE):
        reasons.append(
            "vCard is missing standard name fields (unusual structure)"
        )
        risk += 0.15

    hits = _keyword_hits(payload.lower())
    if hits:
        reasons.append(
            f"Contains scam-associated phrasing: {', '.join(hits[:3])}"
        )
        risk += 0.15 * len(hits)

    reasons.append(UNVERIFIED_NOTE)

    return {
        "risk_score": min(risk, 1.0),
        "reasons": reasons,
        "analysis_status": "unverified",
    }


def analyze_wifi(payload: str) -> dict:
    # WIFI:T:WPA;S:SSID;P:password;;
    reasons = []
    risk = 0.0
    upper = payload.upper()

    if "T:NOPASS" in upper or "T:;" in upper:
        reasons.append(
            "Wi-Fi network has no password. Anyone nearby can join and "
            "watch unencrypted traffic."
        )
        risk += 0.10

    reasons.append(
        "Only join Wi-Fi networks you recognise. This QR cannot prove "
        "the network is genuine."
    )

    return {
        "risk_score": min(risk, 1.0),
        "reasons": reasons,
        "analysis_status": "unverified",
    }