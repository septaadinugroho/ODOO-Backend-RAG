"""Text normalisation, deliberately free of every third-party dependency.

`normalize()` decides whether two strings are "the same word" for this project,
so it has to be identical everywhere: in the request path (app/synonyms.py), in
the offline SQL generator (scripts/gen_sql.py) and in the tests. It lives in its
own module so that `scripts/gen_sql.py` can import the real function using
nothing but the standard library — that script has to run *before* `pip install`,
because it produces the SQL that PostgreSQL loads on its very first boot.

One definition, three callers, no reimplementation to drift out of sync.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold a fragment to its comparison form.

    Lowercase, strip accents, turn punctuation into spaces (so "add-math" and
    "add math" collapse together), and squeeze whitespace. Deliberately keeps
    digits, because levels depend on them: "Form 4" must not become "Form".
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()
