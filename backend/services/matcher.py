# services/matcher.py
# Convenience re-export so the rest of the codebase can import from
# either `services.matcher` or `services.matching_engine`.

from services.matching_engine import (  # noqa: F401
    RetrievalResult,
    MatchingEngine,
    build_engine,
    get_engine,
)
