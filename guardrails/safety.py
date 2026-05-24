"""
safety.py
Lightweight guardrails module for high-stakes AI applications.
Provides three validators:
  1. PII detector — strips emails, phone numbers, NHS/SSN numbers
  2. Jailbreak filter — blocks known prompt injection patterns
  3. Output grounding check — warns if response contains claims
     not grounded in the provided context

This is a basic implementation. For production:
  - Use Guardrails AI (pip install guardrails-ai) for richer validators
  - Use NeMo Guardrails for conversational rail management
  - Add semantic similarity checks for hallucination detection

Docs: https://docs.guardrailsai.com
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── PII Patterns ──────────────────────────────────────────────
PII_PATTERNS = {
    "email":   re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone_uk": re.compile(r"\b(\+44|0)\s?[0-9]{10}\b"),
    "phone_us": re.compile(r"\b(\+1)?\s?\(?[0-9]{3}\)?[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}\b"),
    "nhs_number": re.compile(r"\b[0-9]{3}\s?[0-9]{3}\s?[0-9]{4}\b"),
    "ssn":     re.compile(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b"),
}

# ── Jailbreak Patterns ────────────────────────────────────────
JAILBREAK_PATTERNS = [
    r"ignore (previous|all|your) instructions",
    r"you are now (DAN|an? (unrestricted|evil|unfiltered))",
    r"pretend (you are|to be) (not an? AI|human|without (restrictions|rules))",
    r"disregard (your|all) (guidelines|training|rules|constraints)",
    r"bypass (your|all) (safety|content|ethical) (filters|guidelines|rules)",
    r"act as if you (have no|are without) (restrictions|guidelines|rules)",
    r"jailbreak",
    r"prompt injection",
]
JAILBREAK_RE = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)


class PIIDetectedWarning(Exception):
    """Raised when PII is detected in input."""
    pass


class JailbreakAttemptError(Exception):
    """Raised when a jailbreak pattern is detected."""
    pass


def strip_pii(text: str, raise_on_detect: bool = False) -> str:
    """
    Detect and strip PII from text.
    If raise_on_detect=True, raises PIIDetectedWarning instead of stripping.
    """
    found = []
    result = text
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(result)
        if matches:
            found.append(f"{pii_type} ({len(matches)} instance(s))")
            result = pattern.sub(f"[{pii_type.upper()}_REDACTED]", result)

    if found:
        msg = f"PII detected and redacted: {', '.join(found)}"
        logger.warning(msg)
        if raise_on_detect:
            raise PIIDetectedWarning(msg)

    return result


def check_jailbreak(text: str) -> None:
    """
    Check input for jailbreak/prompt injection patterns.
    Raises JailbreakAttemptError if detected.
    """
    match = JAILBREAK_RE.search(text)
    if match:
        logger.error(f"Jailbreak attempt detected: '{match.group()}'")
        raise JailbreakAttemptError(
            "Input contains prompt injection pattern. Request blocked. "
            f"Matched: '{match.group()}'"
        )


def check_output_grounding(response: str, context: str, min_overlap: float = 0.1) -> bool:
    """
    Basic grounding check: verifies the response is broadly consistent
    with the retrieved context by checking word overlap.

    Returns True if grounded, False if likely hallucinating.
    This is a heuristic — not a semantic check. For production use
    a cross-encoder or NLI model for proper hallucination detection.
    """
    if not context.strip():
        return True   # No context to check against

    response_words = set(response.lower().split())
    context_words  = set(context.lower().split())

    # Remove common stop words for cleaner signal
    stop_words = {"the","a","an","is","are","was","were","to","of","and","in","that","it","for"}
    response_words -= stop_words
    context_words  -= stop_words

    if not response_words:
        return True

    overlap = len(response_words & context_words) / len(response_words)
    grounded = overlap >= min_overlap

    if not grounded:
        logger.warning(
            f"Output grounding check FAILED: {overlap:.0%} word overlap with context. "
            f"Response may contain hallucinated content. "
            f"Review before using in production."
        )

    return grounded


def validate_input(text: str, strip_pii: bool = True) -> str:
    """
    Run all input validators in sequence.
    Returns sanitised text or raises on jailbreak detection.
    """
    check_jailbreak(text)
    if strip_pii:
        text = globals()["strip_pii"](text)
    return text


def validate_output(response: str, context: str = "") -> dict:
    """
    Run all output validators.
    Returns a dict with validation results.
    """
    grounded = check_output_grounding(response, context)
    return {
        "grounded": grounded,
        "warning": None if grounded else "Response may not be fully grounded in retrieved context.",
    }
