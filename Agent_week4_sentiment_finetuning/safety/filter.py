import os, re
from typing import Dict, Any
from openai import OpenAI

# --- simple injection heuristics ---
_INJECTION_PATTERNS = [
    r"\bignore (all|previous|earlier) (instructions|prompts)\b",
    r"\boverride\b.*\bsystem\b",
    r"\bdisregard\b.*\brules\b",
    r"\bprint\b.*\bsystem prompt\b",
    r"\bshow\b.*\bconfidential\b",
    r"\breturn\b.*\btool schema\b",
]
_INJ = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def moderate(text: str) -> Dict[str, Any]:
    """Wrap OpenAI moderation. Fail-open on errors."""
    try:
        m = _client.moderations.create(
            model="omni-moderation-latest",
            input=text or ""
        )
        r = m.results[0]
        flagged = bool(getattr(r, "flagged", False))
        cats = getattr(r, "categories", {}) or {}
        # ensure plain dict
        cats = dict(cats) if not isinstance(cats, dict) else cats
        return {"flagged": flagged, "categories": cats}
    except Exception as e:
        return {"flagged": False, "error": str(e)}

def detect_injection(text: str) -> bool:
    return bool(_INJ.search(text or ""))

def guard_query(text: str) -> Dict[str, Any]:
    inj = detect_injection(text)
    mod = moderate(text)
    blocked = inj or mod.get("flagged", False)
    reason = "prompt_injection" if inj else ""
    if mod.get("flagged", False):
        reason = (reason + " moderation").strip()
    return {"blocked": blocked, "reason": reason, "moderation": mod}
