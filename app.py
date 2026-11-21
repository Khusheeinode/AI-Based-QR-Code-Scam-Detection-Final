"""
app.py - QR Shield (Streamlit web app)

Scan Smart. Stay Safe.

Flow:
  1. Upload a QR image or take a photo with the camera.
  2. Detect and decode every QR code (pyzbar, OpenCV as fallback).
  3. Crop the selected QR with perspective correction (QR only,
     never the whole poster / ticket / screenshot).
  4. Payload analysis (Branch B) and CNN image analysis (Branch A)
     run independently: if one fails, the other still works.
  5. Fusion gives the final Low / Medium / High verdict.

Safety rules in this file:
  * Links are never opened automatically.
  * No "open" button is offered for High risk links.
  * Scanned text is never placed inside HTML; it is shown with st.code()
    or markdown-escaped.
  * Uploaded images are processed in memory and not stored.

Run locally:
    python -m streamlit run app.py
"""

import hashlib
import os
import re
import sys
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
import streamlit as st
from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for _sub in ("branch_a_image", "branch_b_payload", "fusion"):
    sys.path.insert(0, os.path.join(BASE_DIR, _sub))

from payload_pipeline import analyze_payload
from fusion import fuse

try:
    import tensorflow as tf
    from gradcam import predict_with_gradcam
    CNN_IMPORT_ERROR = None
except Exception as exc:  # TensorFlow missing or gradcam failed to import
    tf = None
    predict_with_gradcam = None
    CNN_IMPORT_ERROR = type(exc).__name__

CNN_MODEL_PATH = os.path.join(
    BASE_DIR, "branch_a_image", "qr_cnn_model.keras"
)

MAX_IMAGE_SIDE = 2400
MAX_PAYLOAD_DISPLAY = 1000

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="QR Shield - Scan Smart. Stay Safe.",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
.block-container { padding-top: 1.4rem; max-width: 760px; }
#MainMenu, footer { visibility: hidden; }

.qs-hero { text-align: center; padding: 1rem 0 0.4rem; }
.qs-logo { font-size: 2.1rem; font-weight: 800; letter-spacing: .5px; }
.qs-logo span { color: #22D3EE; }
.qs-tagline { font-size: 1.3rem; font-weight: 700; margin-top: .2rem; }
.qs-sub { color: #9FB0CC; max-width: 560px; margin: .5rem auto 0;
          line-height: 1.55; font-size: .98rem; }

.qs-card { background: #111A2E; border: 1px solid #1F2B45; color: #E6EDF7;
           border-radius: 16px; padding: 1rem 1.1rem; margin: .6rem 0; }
.qs-card h4 { margin: 0 0 .3rem; font-size: 1.02rem; }
.qs-card p  { margin: 0; color: #B6C3DA; line-height: 1.5; font-size: .93rem; }

.qs-verdict { display: flex; align-items: center; gap: .9rem; color: #E6EDF7;
              background: #111A2E; border: 1px solid var(--accent);
              border-left: 6px solid var(--accent); border-radius: 16px;
              padding: 1rem 1.1rem; margin-top: .6rem; }
.qs-verdict-icon { font-size: 2rem; line-height: 1; }
.qs-verdict-label { color: var(--accent); font-weight: 700;
                    text-transform: uppercase; font-size: .78rem;
                    letter-spacing: .09em; }
.qs-verdict-head { font-size: 1.12rem; font-weight: 700; line-height: 1.3; }
.qs-pct { margin-left: auto; font-size: 2rem; font-weight: 800;
          color: var(--accent); }

.qs-bar { height: 10px; background: #1F2B45; border-radius: 999px;
          margin: .6rem 0 1rem; overflow: hidden; }
.qs-bar > div { height: 100%; border-radius: 999px; }

.qs-tiles { display: grid; grid-template-columns: repeat(3, 1fr);
            gap: .6rem; margin: .4rem 0 .8rem; }
.qs-tile { background: #111A2E; border: 1px solid #1F2B45; color: #E6EDF7;
           border-radius: 14px; padding: .7rem .5rem; text-align: center; }
.qs-tile-label { color: #9FB0CC; font-size: .74rem; letter-spacing: .04em;
                 text-transform: uppercase; }
.qs-tile-value { font-size: 1.35rem; font-weight: 800; margin-top: .1rem; }

.qs-chip { display: inline-block; background: #162238;
           border: 1px solid #25365A; border-radius: 999px;
           padding: .25rem .75rem; margin: .15rem .3rem .15rem 0;
           font-size: .85rem; color: #D6E2F5; }

.qs-footer { color: #7F90AD; font-size: .8rem; text-align: center;
             margin-top: 1.6rem; line-height: 1.5; }

.stButton > button, .stDownloadButton > button {
    border-radius: 12px; min-height: 2.8rem; }

@media (max-width: 640px) {
    .qs-logo { font-size: 1.8rem; }
    .qs-tagline { font-size: 1.1rem; }
    .qs-verdict { gap: .6rem; padding: .8rem; }
    .qs-pct { font-size: 1.6rem; }
    .qs-tile-value { font-size: 1.1rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------
# FIXED TEXT (nothing here comes from a scanned QR)
# --------------------------------------------------

BANDS = {
    "Low": {
        "color": "#22C55E",
        "icon": "✅",
        "title": "Low risk",
        "headline": "No major threats detected",
        "advice": (
            "No major warning signs were found. This does not guarantee "
            "safety: no tool can. Stay alert, especially before paying "
            "or entering personal details."
        ),
    },
    "Medium": {
        "color": "#F59E0B",
        "icon": "⚠️",
        "title": "Medium risk",
        "headline": "Suspicious. Verify before you continue",
        "advice": (
            "Some warning signs were found. Check where this QR code came "
            "from and confirm it with the sender or shop before opening "
            "links, paying or sharing information."
        ),
    },
    "High": {
        "color": "#EF4444",
        "icon": "⛔",
        "title": "High risk",
        "headline": "Dangerous. Do not open or pay",
        "advice": (
            "Strong warning signs were found. Do not open the link, make "
            "a payment or enter any details. If it came from a message "
            "or poster, report it and delete it."
        ),
    },
}

BLACKLIST_LABELS = {
    "listed": ("🔴", "Listed", "Confirmed match in a phishing blacklist."),
    "not_found": (
        "🟡", "Not found",
        "Not in the blacklist. This is NOT proof that it is safe.",
    ),
    "unavailable": (
        "⚪", "Verification unavailable",
        "The blacklist check could not be completed, so safety is unconfirmed.",
    ),
    "not_applicable": (
        "⚪", "Not applicable", "No blacklist check for this QR type.",
    ),
}

PAYLOAD_TYPE_LABELS = {
    "url": "Website link",
    "upi": "UPI payment",
    "text": "Plain text",
    "wifi": "Wi-Fi network",
    "vcard": "Contact card",
    "contact_action": "Call / SMS / Email",
    "action_uri": "Action link",
    "custom_uri": "App link",
    "json": "JSON data",
    "xml": "XML data",
    "uuid": "Identifier (UUID)",
    "opaque_token": "Token / ID",
}

# --------------------------------------------------
# SAFE TEXT HELPERS
# --------------------------------------------------

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")


def md_escape(text):
    """Escape markdown so scanned text cannot create links or formatting."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def safe_display(text, limit=MAX_PAYLOAD_DISPLAY):
    """Replace control characters and truncate long text."""
    text = "".join(
        ch if (ch.isprintable() or ch in "\n\t") else "?"
        for ch in str(text)
    )
    return text if len(text) <= limit else text[:limit] + " ..."


# --------------------------------------------------
# CNN MODEL (loaded once)
# --------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    if tf is None or predict_with_gradcam is None:
        return None, CNN_IMPORT_ERROR or "TensorFlow is not available"

    if not os.path.exists(CNN_MODEL_PATH):
        return None, "model file not found"

    try:
        return tf.keras.models.load_model(CNN_MODEL_PATH, compile=False), None
    except Exception as exc:
        return None, type(exc).__name__


@st.cache_data(show_spinner=False, ttl=300, max_entries=64)
def cached_payload_analysis(payload):
    return analyze_payload(payload)


# --------------------------------------------------
# QR DETECTION
# --------------------------------------------------

def decode_image_bytes(data):
    """Bytes -> BGR image (downscaled if huge), or None if not an image."""
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

    if img is None:
        return None

    height, width = img.shape[:2]
    scale = MAX_IMAGE_SIDE / max(height, width)

    if scale < 1.0:
        img = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    return img


def _rect_from_polygon(poly):
    x_min, y_min = poly.min(axis=0)
    x_max, y_max = poly.max(axis=0)
    return (float(x_min), float(y_min),
            float(x_max - x_min), float(y_max - y_min))


def _zbar_codes(gray, scale):
    codes = []

    try:
        results = zbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
    except Exception:
        return codes

    for item in results:
        if item.polygon:
            poly = np.array(
                [[p.x, p.y] for p in item.polygon], dtype=np.float32
            ) / scale
        else:
            left, top, width, height = item.rect
            poly = np.array(
                [[left, top], [left + width, top],
                 [left + width, top + height], [left, top + height]],
                dtype=np.float32,
            ) / scale

        codes.append({
            "payload_bytes": bytes(item.data),
            "polygon": poly,
            "rect": _rect_from_polygon(poly),
        })

    return codes


def _opencv_codes(img):
    """Fallback detector when pyzbar finds nothing."""
    codes = []

    try:
        ok, decoded, points, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
    except Exception:
        return codes

    if not ok or points is None:
        return codes

    for text, poly in zip(decoded, points):
        if text:
            poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            codes.append({
                "payload_bytes": text.encode("utf-8", errors="replace"),
                "polygon": poly,
                "rect": _rect_from_polygon(poly),
            })

    return codes


def detect_qr_codes(img):
    """Returns every QR code found, in reading order."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = [(gray, 1.0), (255 - gray, 1.0)]

    if max(gray.shape) <= 1800:
        up = cv2.resize(
            gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC
        )
        variants += [(up, 2.0), (255 - up, 2.0)]

    codes = []

    for image, scale in variants:
        codes = _zbar_codes(image, scale)
        if codes:
            break

    if not codes:
        codes = _opencv_codes(img)

    unique, seen = [], set()

    for code in codes:
        centre = code["polygon"].mean(axis=0)
        key = (code["payload_bytes"], int(centre[0] // 15), int(centre[1] // 15))
        if key not in seen:
            seen.add(key)
            unique.append(code)

    unique.sort(key=lambda c: (c["polygon"].mean(axis=0)[1],
                               c["polygon"].mean(axis=0)[0]))

    return unique


# --------------------------------------------------
# QR CROP (perspective corrected, QR only)
# --------------------------------------------------

def order_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    rect[0] = points[np.argmin(sums)]    # top-left
    rect[1] = points[np.argmin(diffs)]   # top-right
    rect[2] = points[np.argmax(sums)]    # bottom-right
    rect[3] = points[np.argmax(diffs)]   # bottom-left

    return rect


def _add_quiet_zone(crop):
    """
    Adds a plain border in the QR's own light colour, like the training
    images, so no poster/ticket background reaches the CNN.
    """
    pad = max(8, int(round(crop.shape[0] * 0.11)))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    light = gray >= np.percentile(gray, 85)

    if light.any():
        colour = tuple(int(v) for v in crop[light].mean(axis=0))
    else:
        colour = (255, 255, 255)

    return cv2.copyMakeBorder(
        crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=colour
    )


def crop_qr(img, polygon, rect):
    """Perspective-warps the QR; falls back to its bounding box."""
    height, width = img.shape[:2]

    try:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

        if len(points) < 4:
            raise ValueError("not enough corner points")

        hull = cv2.convexHull(points)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True).reshape(-1, 2)

        if len(approx) == 4:
            corners = approx
        else:
            corners = cv2.boxPoints(cv2.minAreaRect(hull))

        quad = order_points(np.asarray(corners, dtype=np.float32))
        tl, tr, br, bl = quad

        side = int(round(max(
            np.linalg.norm(tr - tl), np.linalg.norm(br - bl),
            np.linalg.norm(bl - tl), np.linalg.norm(br - tr),
        )))

        if side < 20:
            raise ValueError("QR too small")

        size = int(np.clip(side, 120, 800))
        target = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(quad, target)
        crop = cv2.warpPerspective(
            img, matrix, (size, size),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
        )

    except Exception:
        x, y, rect_w, rect_h = rect
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1 = min(width, int(x + rect_w))
        y1 = min(height, int(y + rect_h))

        if x1 - x0 < 20 or y1 - y0 < 20:
            return None

        size = int(np.clip(max(x1 - x0, y1 - y0), 120, 800))
        crop = cv2.resize(img[y0:y1, x0:x1], (size, size))

    return _add_quiet_zone(crop)


# --------------------------------------------------
# BRANCH A: CNN (never allowed to break the app)
# --------------------------------------------------

def run_image_analysis(crop):
    """Returns (image_score or None, heatmap or None, note)."""
    if crop is None:
        return None, None, "The QR region could not be cropped."

    model, error = load_cnn_model()

    if model is None:
        return None, None, f"Image analysis unavailable ({error})."

    path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            path = tmp.name

        cv2.imwrite(path, crop)
        score, overlay = predict_with_gradcam(model, path)

        return float(score), overlay, ""

    except Exception as exc:
        return None, None, f"Image analysis failed ({type(exc).__name__})."

    finally:
        if path and os.path.exists(path):
            os.remove(path)


# --------------------------------------------------
# RESULT DISPLAY
# --------------------------------------------------

def _pct(value):
    return int(round(max(0.0, min(1.0, float(value))) * 100))


def render_verdict(result):
    band = result.get("risk_band")
    cfg = BANDS.get(band, BANDS["Medium"])
    pct = _pct(result["final_score"])

    st.markdown(
        f"""
        <div class="qs-verdict" style="--accent:{cfg['color']}">
          <div class="qs-verdict-icon">{cfg['icon']}</div>
          <div>
            <div class="qs-verdict-label">{cfg['title']}</div>
            <div class="qs-verdict-head">{cfg['headline']}</div>
          </div>
          <div class="qs-pct">{pct}%</div>
        </div>
        <div class="qs-bar">
          <div style="width:{pct}%;background:{cfg['color']}"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if band == "High":
        st.error(cfg["advice"])
    elif band == "Medium":
        st.warning(cfg["advice"])
    else:
        st.success(cfg["advice"])


def render_scores(result):
    final_pct = _pct(result["final_score"])
    payload_pct = _pct(result["payload_risk_score"])
    image_value = result.get("image_risk_score")
    image_text = "N/A" if image_value is None else f"{_pct(image_value)}%"

    st.markdown(
        f"""
        <div class="qs-tiles">
          <div class="qs-tile"><div class="qs-tile-label">Final risk</div>
            <div class="qs-tile-value">{final_pct}%</div></div>
          <div class="qs-tile"><div class="qs-tile-label">Content check</div>
            <div class="qs-tile-value">{payload_pct}%</div></div>
          <div class="qs-tile"><div class="qs-tile-label">Image check</div>
            <div class="qs-tile-value">{image_text}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_blacklist_status(payload_result, result):
    status = result.get("blacklist_status")

    if status in BLACKLIST_LABELS:
        return status

    if payload_result.get("blacklisted") is True:
        return "listed"

    checked = payload_result.get("blacklist_checked")

    if checked is True:
        return "not_found"
    if checked is False:
        return "unavailable"

    return "not_applicable"


def render_info_chips(payload_type, status):
    icon, label, _ = BLACKLIST_LABELS[status]
    type_label = PAYLOAD_TYPE_LABELS.get(payload_type, "Other")

    st.markdown(
        f"""
        <span class="qs-chip">Type: {type_label}</span>
        <span class="qs-chip">Blacklist: {icon} {label}</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption(BLACKLIST_LABELS[status][2])


def render_reasons(reasons):
    st.markdown("#### Why this result")

    if not reasons:
        st.write("No specific warning signs were reported.")
        return

    st.markdown("\n".join(f"- {md_escape(r)}" for r in reasons[:12]))


def render_upi_panel(payload_result):
    fields = payload_result.get("fields") or {}

    st.markdown("#### Payment details in this QR")

    payee_id = fields.get("pa") or "Not specified"
    name = fields.get("pn") or "Not specified"
    amount = fields.get("am")
    currency = fields.get("cu") or "Not specified"
    note = fields.get("tn") or "None"

    amount_text = f"{amount}" if amount else "Amount not specified in QR"

    st.markdown(
        "\n".join([
            f"- **Payee UPI ID:** {md_escape(str(payee_id)[:80])}",
            f"- **Payee name:** {md_escape(str(name)[:80])}",
            f"- **Amount:** {md_escape(str(amount_text)[:30])}",
            f"- **Currency:** {md_escape(str(currency)[:10])}",
            f"- **Note:** {md_escape(str(note)[:100])}",
        ])
    )

    st.info(
        "QR Shield never makes or authorises payments. Check the payee "
        "name shown in your UPI app before you pay."
    )


def _openable_url(payload):
    """Returns (url, host) only for a plain, well-formed http(s) link."""
    url = payload.strip()

    if len(url) > 2000 or re.search(r"[\s\x00-\x1f]", url):
        return None, None

    try:
        parsed = urlparse(url)
        host = parsed.hostname
    except ValueError:
        return None, None

    if parsed.scheme not in ("http", "https") or not host:
        return None, None

    return url, host


def render_link_panel(payload, payload_type, band):
    if payload_type != "url":
        return

    url, host = _openable_url(payload)

    st.markdown("#### Link details")

    if host is None:
        st.warning(
            "This link is unusual or malformed, so QR Shield will not "
            "offer an Open button."
        )
        return

    scheme_ok = urlparse(url).scheme == "https"
    st.markdown(
        f"- **Website:** {md_escape(host)}\n"
        f"- **Secure connection (HTTPS):** {'Yes' if scheme_ok else 'No'}"
    )

    if "xn--" in host or any(ord(ch) > 127 for ch in host):
        st.warning(
            "This web address uses special characters that can imitate a "
            "trusted site (look-alike domain). Be extra careful."
        )

    if band == "High":
        st.error(
            "For your safety, QR Shield does not offer an Open button for "
            "High-risk links."
        )
        return

    key = "ack_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]

    if band == "Medium":
        allowed = st.checkbox(
            "I understand this link may be unsafe and I want to open it.",
            key=key,
        )
    else:
        allowed = True

    if allowed and hasattr(st, "link_button"):
        st.link_button("Open link in a new tab", url)
    elif allowed:
        st.caption("Copy the link above and open it in your browser.")


def render_image_panel(crop, overlay, image_score, note):
    st.markdown("#### Image analysis")

    if note:
        st.info(note)

    if crop is not None:
        st.image(
            cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            caption="QR code cropped from your image (only this part is checked)",
            width=240,
        )

    if overlay is not None:
        st.image(
            overlay,
            caption=(
                "Grad-CAM heatmap: warmer areas pushed the AI towards "
                "'tampered'. If the image check is low, it is not evidence "
                "of tampering."
            ),
            width=240,
        )


def render_result(payload, non_utf8, payload_result, result,
                  crop, overlay, image_note):
    st.divider()
    render_verdict(result)
    render_scores(result)

    payload_type = result.get("payload_type") or payload_result.get("payload_type")
    status = get_blacklist_status(payload_result, result)

    render_info_chips(payload_type, status)

    if non_utf8:
        st.warning(
            "This QR contains bytes that are not valid text. The analysed "
            "text may differ slightly from the real content."
        )

    if payload_result.get("analysis_status") in ("unverified", "degraded"):
        st.info(
            "This kind of content cannot be fully verified automatically, "
            "so it is not treated as safe."
        )

    st.markdown("#### What this QR contains")
    st.code(safe_display(payload), language=None)
    st.caption("Shown as plain text. QR Shield never opens or runs it for you.")

    render_reasons(result.get("reasons") or [])

    if payload_type == "upi":
        render_upi_panel(payload_result)

    render_link_panel(payload, payload_type, result.get("risk_band"))

    with st.expander("Image analysis details"):
        render_image_panel(crop, overlay, result.get("image_risk_score"), image_note)


# --------------------------------------------------
# SCAN FLOW
# --------------------------------------------------

def process_image(data):
    img = decode_image_bytes(data)

    if img is None:
        st.error(
            "This file could not be read as an image. Please upload a "
            "PNG or JPG picture."
        )
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Your image", width=260)

    with st.spinner("Looking for QR codes..."):
        codes = detect_qr_codes(img)

    if not codes:
        st.error("No readable QR code was found in this image.")
        st.markdown(
            "- Make sure the QR code is sharp and well lit\n"
            "- Keep the whole QR code inside the picture\n"
            "- Try a closer photo, or crop the screenshot"
        )
        return

    index = 0

    if len(codes) > 1:
        st.warning(f"{len(codes)} QR codes were found. Choose one to analyse.")

        def _label(i):
            text = codes[i]["payload_bytes"].decode("utf-8", errors="replace")
            return f"QR {i + 1}: {safe_display(text, 40)}"

        index = st.selectbox(
            "QR code to analyse", list(range(len(codes))), format_func=_label
        )

    code = codes[index]
    raw = code["payload_bytes"]

    try:
        payload = raw.decode("utf-8")
        non_utf8 = False
    except UnicodeDecodeError:
        payload = raw.decode("utf-8", errors="replace")
        non_utf8 = True

    # Branch B: payload analysis
    try:
        with st.spinner("Checking what the QR code contains..."):
            payload_result = cached_payload_analysis(payload)
    except Exception:
        st.error(
            "The content of this QR code could not be analysed, so no "
            "verdict is shown. Treat it as unverified and do not open or "
            "pay using it."
        )
        st.code(safe_display(payload), language=None)
        return

    # Branch A: image analysis (optional)
    crop = crop_qr(img, code["polygon"], code["rect"])

    with st.spinner("Checking the QR image..."):
        image_score, overlay, image_note = run_image_analysis(crop)

    # Fusion
    try:
        result = fuse(image_score, payload_result)
    except Exception:
        st.error(
            "A risk verdict could not be produced for this QR code. "
            "Treat it as unverified."
        )
        return

    render_result(
        payload, non_utf8, payload_result, result, crop, overlay, image_note
    )


def scan_page():
    mode = st.radio(
        "How do you want to scan?",
        ["Upload image", "Use camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    data = None

    if mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload a QR code image (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            data = uploaded.getvalue()
    else:
        st.caption(
            "Hold the QR code steady, fill the frame, then take the photo. "
            "Your browser will ask for camera permission."
        )
        photo = st.camera_input("Take a photo of the QR code")
        if photo is not None:
            data = photo.getvalue()

    if data is not None:
        process_image(data)
    else:
        st.markdown(
            """
            <div class="qs-card"><h4>Not sure about a QR code?</h4>
            <p>Upload a screenshot or photo, and QR Shield will tell you if
            it looks safe, suspicious or dangerous, and why.</p></div>
            """,
            unsafe_allow_html=True,
        )


def how_it_works_page():
    st.markdown(
        """
        <div class="qs-card"><h4>1. Content check</h4>
        <p>Reads what the QR code contains: a website link, a UPI payment,
        text, Wi-Fi details or a contact card. It looks for phishing
        signs, look-alike domains, link shorteners, suspicious payment
        requests and known phishing blacklists.</p></div>

        <div class="qs-card"><h4>2. Image check</h4>
        <p>An AI model checks the QR picture itself for signs of tampering,
        such as a sticker pasted over the real code. A heatmap shows which
        part of the image influenced the result.</p></div>

        <div class="qs-card"><h4>3. Final verdict</h4>
        <p>Both checks are combined into Low, Medium or High risk. A picture
        that only looks unusual can never make a QR High risk on its own,
        and a confirmed blacklist match always does.</p></div>
        """,
        unsafe_allow_html=True,
    )


def about_page():
    st.markdown(
        """
        <div class="qs-card"><h4>About QR Shield</h4>
        <p>An AI-based QR code scam detection project built by students of
        Usha Mittal Institute of Technology, SNDT Women's University,
        Mumbai (Artificial Intelligence department).</p></div>

        <div class="qs-card"><h4>Privacy</h4>
        <p>Your image is processed in memory and is not saved by QR Shield.
        For website links, the link may be sent to the PhishTank service to
        check it against known phishing sites.</p></div>

        <div class="qs-card"><h4>Please note</h4>
        <p>Results are advice, not a guarantee. QR Shield never opens links
        automatically and never makes or approves payments. A Low result
        does not prove a QR code is safe.</p></div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    st.markdown(
        """
        <div class="qs-hero">
          <div class="qs-logo">🛡️ QR <span>Shield</span></div>
          <div class="qs-tagline">Scan Smart. Stay Safe.</div>
          <div class="qs-sub">Check a QR code for phishing links, fake
          payment requests and tampering, before you open it or pay.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scan_tab, how_tab, about_tab = st.tabs(
        ["🔍 Scan QR", "⚙️ How it works", "ℹ️ About"]
    )

    with scan_tab:
        scan_page()

    with how_tab:
        how_it_works_page()

    with about_tab:
        about_page()

    st.markdown(
        """
        <div class="qs-footer">QR Shield gives advice only and cannot
        guarantee safety.<br>Never share OTPs or PINs, and never scan to
        receive money.</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()