"""
blacklist_check.py

Phishing blacklist lookup using two independent sources:

1. OpenPhish community feed (no key needed, downloaded once an hour and
   kept in memory, so each scan is instant).
2. PhishTank check-URL API (optional key).

A URL counts as "checked" if at least one source answered. A URL is
"in_blacklist" only if a source confirms it. Not being listed is NEVER
proof that a URL is safe.

check_phishtank() keeps its old name and return keys so payload_pipeline.py
does not need to change.
"""

import threading
import time
from urllib.parse import urlsplit

import requests

PHISHTANK_CHECK_URL = "https://checkurl.phishtank.com/checkurl/"
OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"
USER_AGENT = "phishtank/qrshield-student"

FEED_TTL = 60 * 60            # refresh the feed every hour
FEED_STALE_LIMIT = 24 * 3600  # use an old copy for up to a day if refresh fails
FEED_RETRY_AFTER = 120        # after a failed download, wait before retrying
FEED_MAX_BYTES = 10_000_000
PHISHTANK_RETRY_AFTER = 120

_lock = threading.Lock()
_feed = {"urls": frozenset(), "fetched": 0.0, "retry_at": 0.0}
_pt_state = {"retry_at": 0.0}


def _is_true(value):
    """Accept only explicit positive values."""
    return value is True or (
        isinstance(value, str)
        and value.strip().lower() in ("y", "true")
    )


def _normalize(url):
    """host/path?query, lower-case host, no scheme, no 'www.', no fragment."""
    try:
        parts = urlsplit(url.strip())
        host = (parts.hostname or "").lower()
    except ValueError:
        return None

    if not host:
        return None

    if host.startswith("www."):
        host = host[4:]

    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


# --------------------------------------------------
# SOURCE 1: OpenPhish feed
# --------------------------------------------------

def _load_openphish_feed():
    """Returns a set of normalized URLs, or None if no feed is available."""
    now = time.time()

    with _lock:
        age = now - _feed["fetched"]

        if _feed["urls"] and age < FEED_TTL:
            return _feed["urls"]

        if now < _feed["retry_at"]:
            return _feed["urls"] if _feed["urls"] and age < FEED_STALE_LIMIT else None

        try:
            response = requests.get(
                OPENPHISH_FEED_URL,
                headers={"User-Agent": USER_AGENT},
                timeout=8,
            )
            response.raise_for_status()

            if len(response.content) > FEED_MAX_BYTES:
                raise ValueError("Feed is unexpectedly large.")

            entries = set()
            for line in response.text.splitlines():
                key = _normalize(line)
                if key:
                    entries.add(key)

            if not entries:
                raise ValueError("Feed was empty.")

            _feed["urls"] = frozenset(entries)
            _feed["fetched"] = now
            _feed["retry_at"] = 0.0
            return _feed["urls"]

        except (requests.RequestException, ValueError):
            _feed["retry_at"] = now + FEED_RETRY_AFTER
            if _feed["urls"] and age < FEED_STALE_LIMIT:
                return _feed["urls"]
            return None


def _check_openphish(url):
    feed = _load_openphish_feed()

    if feed is None:
        return {"checked": False, "listed": False,
                "error": "OpenPhish feed unavailable."}

    key = _normalize(url)
    return {"checked": True, "listed": bool(key and key in feed), "error": None}


# --------------------------------------------------
# SOURCE 2: PhishTank API
# --------------------------------------------------

def _check_phishtank_api(url, api_key, timeout):
    if time.time() < _pt_state["retry_at"]:
        return {"checked": False, "listed": False,
                "error": "PhishTank recently failed; skipped."}

    data = {"url": url, "format": "json"}

    if api_key:
        data["app_key"] = api_key

    try:
        response = requests.post(
            PHISHTANK_CHECK_URL,
            data=data,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()

        if "json" not in response.headers.get("content-type", "").lower():
            raise ValueError("PhishTank returned a non-JSON response.")

        body = response.json()

        if not isinstance(body, dict):
            raise ValueError("Invalid JSON response structure.")

        results = body.get("results")

        if not isinstance(results, dict):
            raise ValueError("Missing or invalid results field.")

        if "in_database" not in results or "valid" not in results:
            # URL is simply not in the database
            if _is_true(results.get("in_database")) is False and \
               results.get("in_database") is not None:
                return {"checked": True, "listed": False, "error": None}
            raise ValueError("Missing blacklist verification fields.")

        listed = _is_true(results["in_database"]) and _is_true(results["valid"])
        return {"checked": True, "listed": listed, "error": None}

    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
        _pt_state["retry_at"] = time.time() + PHISHTANK_RETRY_AFTER
        return {"checked": False, "listed": False, "error": str(exc)}


# --------------------------------------------------
# PUBLIC FUNCTION (name kept for payload_pipeline.py)
# --------------------------------------------------

def check_phishtank(url: str, api_key: str = None, timeout: int = 6) -> dict:
    """
    Three possible states:
    Confirmed match: checked=True,  in_blacklist=True
    Not listed:      checked=True,  in_blacklist=False
    Unknown:         checked=False, in_blacklist=False  (no source answered)
    """
    results = {
        "OpenPhish": _check_openphish(url),
    }

    # A confirmed match needs no second opinion.
    if not results["OpenPhish"]["listed"]:
        results["PhishTank"] = _check_phishtank_api(url, api_key, timeout)

    checked_sources = [n for n, r in results.items() if r["checked"]]
    listed_sources = [n for n, r in results.items() if r["listed"]]
    errors = [r["error"] for r in results.values() if r["error"]]

    return {
        "in_blacklist": bool(listed_sources),
        "checked": bool(checked_sources),
        "error": None if checked_sources else "; ".join(errors) or None,
        "sources_checked": checked_sources,
        "sources_listed": listed_sources,
    }
