"""
The layered defense pipeline that sits between the user and the RAG chain.

Each layer returns a Verdict. The orchestrator in app.py runs them in
COST ORDER (cheapest / free gates first) and short-circuits on the first
block — so an off-topic request never reaches the paid generation call.

Layers implemented here:
  Layer 3a — scan_input()        : free regex + length checks
  Layer 2  — retrieve_and_gate() : cheap embedding + local FAISS relevance check
  Layer 3b — sanitize_output()   : strip any code blocks that slip through

(Layer 1 = hardened prompt and Layer 4 = rate limit / token cap live in app.py,
because they're wired into the LLM and the Flask route directly.)
"""

import re
import logging
from dataclasses import dataclass

import config

# ─── Set up a simple structured logger for blocked attempts ──────────────────
# This is your hook for the monitoring dashboard later: every block is logged
# with the layer + reason, so you can chart attack patterns over time.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
_log = logging.getLogger("defense")

# Pre-compile patterns ONCE at import for speed (not on every request).
_BLOCK_RE = [re.compile(p, re.IGNORECASE) for p in config.BLOCKLIST_PATTERNS]
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class Verdict:
    """Result of a defense layer. `reason` is for logs; `message` is for users."""
    allowed: bool
    reason: str = ""      # e.g. "layer2:off_topic:dist=1.42" — internal, logged
    message: str = ""     # user-facing text shown when blocked


PASS = Verdict(allowed=True)


# ─── Layer 3a: Input guardrails (FREE — runs first) ──────────────────────────
def scan_input(question: str) -> Verdict:
    """Reject over-long inputs and obvious code/computation requests via regex."""
    if len(question) > config.MAX_INPUT_CHARS:
        return Verdict(
            False,
            "layer3:input_too_long",
            config.REFUSAL_MESSAGE,
        )
    for rx in _BLOCK_RE:
        if rx.search(question):
            return Verdict(
                False,
                f"layer3:blocked_pattern:{rx.pattern[:30]}",
                config.REFUSAL_MESSAGE,
            )
    return PASS


# ─── Layer 2: Semantic relevance gate (CHEAP — no generation) ────────────────
def retrieve_and_gate(question: str, vectorstore):
    """
    Embed the query and check its distance to the nearest knowledge chunk.
    Off-topic queries sit far from everything in the resume.

    Returns (Verdict, docs). On PASS, `docs` are the retrieved chunks — reused
    by the generation step so we never embed the same query twice.
    """
    results = vectorstore.similarity_search_with_score(question, k=3)
    if not results:
        return Verdict(False, "layer2:no_match", config.REFUSAL_MESSAGE), []

    best_distance = min(score for _, score in results)   # lower = more relevant
    if best_distance > config.MAX_DISTANCE:
        return (
            Verdict(False, f"layer2:off_topic:dist={best_distance:.2f}",
                    config.REFUSAL_MESSAGE),
            [],
        )

    docs = [doc for doc, _ in results]
    return PASS, docs


# ─── Layer 3b: Output guardrails ─────────────────────────────────────────────
def sanitize_output(answer: str) -> str:
    """
    Belt-and-suspenders: with Layers 1-2 in place almost no code should ever
    reach output, but if the model emits a markdown code block anyway, strip it.
    """
    return _CODE_FENCE_RE.sub("[code removed]", answer)


# ─── Monitoring hook ─────────────────────────────────────────────────────────
def log_block(question: str, verdict: Verdict, client_ip: str = "-"):
    """Record a blocked request. Wire this to a DB/dashboard when you scale."""
    _log.info(
        "BLOCKED ip=%s reason=%s | query=%r",
        client_ip, verdict.reason, question[:120],
    )