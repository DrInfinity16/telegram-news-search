import unicodedata

MIN_SINGLE_TOKEN_CHARS = 4


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        c for c in normalized if unicodedata.category(c) != "Mn"
    ).lower()


def norm_for_search(text: str | None) -> str:
    """Accent/case-folded text stored for SQLite lexical retrieval.

    Values are space-padded so short-token boundary checks can use
    `LIKE '% token %'` without ambiguous substring matches.
    """
    if not text:
        return " "
    return f" {normalize_text(text)} "


def token_in_normalized_text(token: str, norm_text: str) -> bool:
    """Return True when token appears in already-normalized text.

    Short tokens (< MIN_SINGLE_TOKEN_CHARS) require word boundaries so
    acronyms like "IA" cannot match inside unrelated substrings.
    Longer tokens may use substring matching for recall (economia/economia).
    """
    if not token or not norm_text:
        return False
    n = normalize_text(token)
    if not n:
        return False
    if len(n) < MIN_SINGLE_TOKEN_CHARS:
        return f" {n} " in f" {norm_text} "
    return n in norm_text


def phrase_in_normalized_text(phrase: str, norm_text: str) -> bool:
    """True when phrase appears, or meaningful words of phrase all appear."""
    if not phrase or not norm_text:
        return False
    nphrase = normalize_text(phrase)
    if nphrase and nphrase in norm_text:
        return True
    words = [normalize_text(w) for w in phrase.split()]
    words = [w for w in words if len(w) >= MIN_SINGLE_TOKEN_CHARS]
    if len(words) > 1:
        return all(token_in_normalized_text(w, norm_text) for w in words)
    return False
