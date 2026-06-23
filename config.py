"""
Central configuration for the layered defense system.

Keeping every threshold, limit, and pattern here means you can tune the
defense without touching pipeline logic — and it documents, in one place,
exactly what your chatbot is and isn't allowed to do.
"""

# ─── Layer 1: Persona & refusal (hardened prompt) ────────────────────────────
PERSONA_NAME = "Resume Assistant"

ALLOWED_DOMAIN = (
    "the professional background, skills, work experience, education, and "
    "projects of the candidate described in the knowledge base"
)

# The single user-facing message returned whenever ANY layer blocks a request.
# Keep it consistent so attackers can't fingerprint which layer caught them.
REFUSAL_MESSAGE = (
    "I'm a resume assistant — I can only answer questions about the candidate's "
    "background, skills, experience, education, and projects. I can't help with "
    "coding, math, or general-knowledge requests."
)


# ─── Layer 2: Semantic relevance gate ────────────────────────────────────────
# FAISS returns an L2 distance: SMALLER = more semantically similar.
# Off-topic queries (coding, math, trivia) land far from every resume chunk.
# This value is DATA-DEPENDENT — run calibrate.py to pick the right number
# for your knowledge base + embedding model before trusting it.
MAX_DISTANCE = 0.76   # starting point only — CALIBRATE THIS


# ─── Layer 3: Input / output guardrails ──────────────────────────────────────
# Patterns that strongly signal someone trying to use the bot as a general
# coding / computation engine rather than a resume Q&A tool. Cheap regex,
# runs before any API call.
BLOCKLIST_PATTERNS = [
    r"\bimport\s+\w+",                        # import pandas
    r"\bfrom\s+\w+\s+import\b",               # from x import y
    r"\bdef\s+\w+\s*\(",                      # def foo(
    r"\bclass\s+\w+\s*[\(:]",                 # class Foo:
    r"\bfor\s*\(.*;.*;.*\)",                  # C-style for loops
    r"\b(select|insert|update|delete)\b.+\bfrom\b",   # SQL
    r"```",                                   # markdown code fences in the input
    r"\b(write|generate|create|build|debug|fix|refactor|solve|compute|calculate)\b"
    r".{0,40}\b(code|program|script|function|app|algorithm|equation|query)\b",
    r"\bleetcode\b|\bcompile\b|\bstack\s*trace\b",
]

# Reject very long inputs before they ever get embedded (cheap length gate).
MAX_INPUT_CHARS = 600


# ─── Layer 4: Infrastructure & cost controls ─────────────────────────────────
# Cap generation length. Sized to fit a real resume answer comfortably, NOT
# artificially tiny (that would truncate legitimate users too). The freeloader
# is already stopped at Layers 2-3; this is just a runaway-cost backstop.
MAX_OUTPUT_TOKENS = 512

# Per-client-IP rate limits (flask-limiter syntax).
RATE_LIMIT = "20 per minute"
DAILY_LIMIT = "300 per day"