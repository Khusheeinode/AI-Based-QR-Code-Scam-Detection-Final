"""
fusion.py

Combines Branch A (CNN image risk) and Branch B (payload risk)
into a single explainable Low/Medium/High verdict.

Rules
-----
- Image risk can raise risk but cannot lower payload risk.
- A confirmed blacklist match forces High Risk.
- Image doubt alone can NEVER produce High Risk.
- When the decoded content looks clean (payload risk below Low),
  image doubt alone cannot even leave the Low band. Real QR codes with
  logos, gradients, rounded dots or dark mode (Google Pay, PhonePe,
  Instagram, scanner apps) can confuse the image model, and what a person
  would open or pay is the decoded content, which is analysed separately.
- Official brand QR false-positive protection limits the image score for
  recognised brand domains.
"""

import math

IMAGE_WEIGHT = 0.4
PAYLOAD_WEIGHT = 0.6

HARD_OVERRIDE_BLACKLIST_SCORE = 0.95

LOW_UPPER = 0.3
MEDIUM_UPPER = 0.7

# Highest score that image doubt alone may reach when content looks clean.
VISUAL_ONLY_CAP = LOW_UPPER - 0.001


def validate_score(value, name):
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a valid number.")

    if not math.isfinite(score):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return score


def band(score: float) -> str:
    if score < LOW_UPPER:
        return "Low"
    elif score < MEDIUM_UPPER:
        return "Medium"
    return "High"


def blacklist_status(payload_result: dict) -> str:
    if payload_result.get("blacklisted") is True:
        return "listed"

    checked = payload_result.get("blacklist_checked")

    if checked is True:
        return "not_found"
    if checked is False:
        return "unavailable"

    return "not_applicable"


def _is_trusted_brand_qr(payload_result, payload_score):
    return (
        payload_result.get("brand_trust") in ("clean_root", "brand_path")
        and payload_score <= 0.20
        and payload_result.get("blacklisted") is not True
    )


def fuse(image_risk_score, payload_result: dict) -> dict:

    if not isinstance(payload_result, dict):
        raise TypeError("payload_result must be a dictionary.")

    if "risk_score" not in payload_result:
        raise KeyError("payload_result is missing 'risk_score'.")

    payload_score = validate_score(
        payload_result["risk_score"], "Payload risk score"
    )

    blacklisted = payload_result.get("blacklisted") is True
    trusted_brand_qr = _is_trusted_brand_qr(payload_result, payload_score)

    visual_doubt_only = False   # image unsure, content clean
    visual_capped = False       # ...and the cap actually changed the score

    # ----------------------------------------------
    # IMAGE ANALYSIS UNAVAILABLE
    # ----------------------------------------------
    if image_risk_score is None:
        image_score = None
        base_score = payload_score

    # ----------------------------------------------
    # IMAGE ANALYSIS AVAILABLE
    # ----------------------------------------------
    else:
        image_score = validate_score(image_risk_score, "Image risk score")

        effective_image_score = image_score

        if trusted_brand_qr:
            effective_image_score = min(image_score, 0.25)

        weighted_score = (
            IMAGE_WEIGHT * effective_image_score
            + PAYLOAD_WEIGHT * payload_score
        )

        # Image cannot reduce payload risk
        base_score = max(payload_score, weighted_score)

        # Clean content + image doubt only: stay in the Low band.
        if payload_score < LOW_UPPER and not blacklisted:

            visual_doubt_only = image_score >= 0.5

            ceiling = max(payload_score, VISUAL_ONLY_CAP)

            if base_score > ceiling:
                base_score = ceiling
                visual_capped = True

    final_score = base_score
    override_triggered = False

    # ----------------------------------------------
    # BLACKLIST OVERRIDE
    # ----------------------------------------------
    if blacklisted:
        final_score = max(final_score, HARD_OVERRIDE_BLACKLIST_SCORE)
        override_triggered = True

    # ----------------------------------------------
    # IMAGE-ONLY HIGH RISK PROTECTION
    # ----------------------------------------------
    elif payload_score < MEDIUM_UPPER:
        final_score = min(final_score, MEDIUM_UPPER - 0.001)

    # ----------------------------------------------
    # REASONS
    # ----------------------------------------------
    reasons = list(payload_result.get("reasons") or [])

    if image_score is None:
        reasons.append(
            "Image analysis was unavailable; "
            "this verdict is based on payload analysis only."
        )

    elif visual_doubt_only:
        reasons.append(
            f"The image check was unsure ({round(image_score * 100)}%), but "
            "the QR content looked clean, so this did not raise the risk. "
            "Logos, gradients and styled designs can confuse the image check."
        )

    elif image_score >= 0.5:
        reasons.append(
            f"Visual QR anomaly detected (CNN score: {image_score:.2f})"
        )

    if trusted_brand_qr:
        reasons.append(
            "Recognized official brand domain detected. "
            "The image anomaly score was limited to prevent "
            "false positives on genuine branded QR designs."
        )

    if override_triggered:
        reasons.append("High Risk: URL matched a verified blacklist entry.")

    elif (
        not visual_capped
        and payload_score < MEDIUM_UPPER
        and base_score >= MEDIUM_UPPER
    ):
        reasons.append("Visual anomaly alone was capped at Medium Risk.")

    reasons = list(dict.fromkeys(reasons))

    return {
        "final_score": round(final_score, 3),
        "risk_band": band(final_score),
        "image_available": image_score is not None,
        "image_risk_score": (
            round(image_score, 3) if image_score is not None else None
        ),
        "payload_risk_score": round(payload_score, 3),
        "payload_type": payload_result.get("payload_type"),
        "reasons": reasons,
        "blacklist_override": override_triggered,
        "blacklist_checked": payload_result.get("blacklist_checked"),
        "blacklist_status": blacklist_status(payload_result),
    }