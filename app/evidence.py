"""Shared retrieval/ranking evidence helpers.

Stopwords are true grammatical/function words with almost no topical
information. Weak/contextual terms may support relevance but must not
create it alone. Content-bearing topic terms are intentionally not listed
as stopwords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from app.text import normalize_text

if TYPE_CHECKING:
    from app.models import Article
    from app.query_intent import QueryIntent

_WORD_STRIP_RE = re.compile(
    r"^[^\wÁÉÍÓÚÜÑáéíóúüñ]+|[^\wÁÉÍÓÚÜÑáéíóúüñ]+$",
    re.UNICODE,
)

# Conservative Spanish function-word set (normalized).
SPANISH_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "así",
        "asi",
        "aun",
        "aunque",
        "bajo",
        "bien",
        "cada",
        "casi",
        "como",
        "con",
        "contra",
        "cual",
        "cuando",
        "cuanto",
        "de",
        "del",
        "demás",
        "demas",
        "desde",
        "donde",
        "dos",
        "durante",
        "e",
        "el",
        "ella",
        "ellas",
        "ello",
        "ellos",
        "en",
        "entonces",
        "entre",
        "era",
        "eran",
        "eres",
        "es",
        "esa",
        "esas",
        "ese",
        "eso",
        "esos",
        "esta",
        "estaba",
        "están",
        "estan",
        "estar",
        "estas",
        "este",
        "esto",
        "estos",
        "estoy",
        "fue",
        "fueron",
        "ha",
        "había",
        "habia",
        "han",
        "hasta",
        "hay",
        "incluso",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "más",
        "mas",
        "me",
        "mi",
        "mis",
        "mucho",
        "muchos",
        "muy",
        "ni",
        "no",
        "nos",
        "nosotros",
        "nuestra",
        "nuestro",
        "o",
        "os",
        "otra",
        "otras",
        "otro",
        "otros",
        "para",
        "pero",
        "poco",
        "por",
        "porque",
        "pues",
        "que",
        "quien",
        "quienes",
        "se",
        "segun",
        "según",
        "ser",
        "si",
        "sido",
        "sin",
        "sino",
        "sobre",
        "solo",
        "son",
        "su",
        "sus",
        "también",
        "tambien",
        "tan",
        "tanto",
        "te",
        "tiene",
        "tienen",
        "todo",
        "todos",
        "tras",
        "tu",
        "tus",
        "un",
        "una",
        "unas",
        "uno",
        "unos",
        "vosotros",
        "y",
        "ya",
        "yo",
    }
)

# Weak/context-dependent terms: may support relevance, never create it alone.
WEAK_CONTEXT_TOKENS = frozenset(
    {
        "actividad",
        "actividades",
        "area",
        "areas",
        "apoyo",
        "ayuda",
        "campo",
        "campos",
        "caso",
        "casos",
        "ciudad",
        "ciudades",
        "ciudadanos",
        "comunidad",
        "comunidades",
        "contexto",
        "datos",
        "dia",
        "dias",
        "estado",
        "estados",
        "familia",
        "familias",
        "forma",
        "general",
        "gobierno",
        "gobiernos",
        "grupo",
        "grupos",
        "habitant",
        "habitantes",
        "hecho",
        "hechos",
        "informacion",
        "institucion",
        "instituciones",
        "internacional",
        "internacionales",
        "local",
        "locales",
        "lugar",
        "lugares",
        "lucha",
        "mercado",
        "mercados",
        "momento",
        "mundo",
        "nacional",
        "nacionales",
        "noticia",
        "noticias",
        "organizacion",
        "organizaciones",
        "pais",
        "paises",
        "parte",
        "personas",
        "persona",
        "poblacion",
        "politica",
        "politicas",
        "politico",
        "politicos",
        "proceso",
        "procesos",
        "proyecto",
        "proyectos",
        "publico",
        "publica",
        "region",
        "regiones",
        "sector",
        "sectores",
        "situacion",
        "social",
        "sociales",
        "sistema",
        "sistemas",
        "trabajo",
        "trabajos",
        "veces",
        "vez",
        "ano",
        "anos",
        "mexico",
        "mexicana",
        "mexicanas",
        "mexicanos",
        "mujeres",
        "mujer",
    }
)


def clean_token(raw: str) -> str:
    token = _WORD_STRIP_RE.sub("", raw.strip())
    if not token:
        return ""
    return token


def is_stopword(token: str) -> bool:
    if not token:
        return False
    return normalize_text(token) in SPANISH_STOPWORDS


def is_weak_context(token: str) -> bool:
    if not token:
        return False
    return normalize_text(token) in WEAK_CONTEXT_TOKENS


def is_informative_token(token: str) -> bool:
    """Token usable as retrieval/ranking evidence."""
    if not token:
        return False
    return not is_stopword(token)


def phrase_tokens(phrase: str) -> list[str]:
    tokens: list[str] = []
    for raw in phrase.split():
        token = clean_token(raw)
        if token:
            tokens.append(token)
    return tokens


def informative_phrase_tokens(phrase: str) -> list[str]:
    return [t for t in phrase_tokens(phrase) if is_informative_token(t)]


def phrase_has_informative_token(phrase: str) -> bool:
    return bool(informative_phrase_tokens(phrase))


def phrase_has_strong_token(phrase: str) -> bool:
    """Phrase contains at least one informative token that is not weak-only."""
    return any(is_informative_token(t) and not is_weak_context(t) for t in phrase_tokens(phrase))


def informative_tokens_from_phrases(phrases: Iterable[str]) -> list[str]:
    seen_norm: set[str] = set()
    result: list[str] = []
    for phrase in phrases:
        for token in phrase_tokens(phrase):
            if not is_informative_token(token):
                continue
            n = normalize_text(token)
            if not n or n in seen_norm:
                continue
            seen_norm.add(n)
            result.append(token)
    return result


def filter_meaningful_matches(raw_matches: Iterable[str]) -> list[str]:
    """Drop stopwords; never expose stopword tokens in match evidence.

    Multi-word phrases without stopwords are kept intact (e.g. "personas
    desaparecidas"). Phrases that contain stopwords are expanded to their
    informative tokens only (e.g. "guerra en Rusia" -> "guerra", "Rusia").
    """
    result: list[str] = []
    seen_norm: set[str] = set()
    for match in raw_matches:
        if not match or not match.strip():
            continue
        words = phrase_tokens(match)
        if not words:
            continue
        if len(words) == 1:
            if is_stopword(words[0]):
                continue
            tokens_out = [words[0]]
        else:
            if not phrase_has_informative_token(match):
                continue
            if any(is_stopword(w) for w in words):
                tokens_out = informative_phrase_tokens(match)
            else:
                tokens_out = [match.strip()]
        for item in tokens_out:
            n = normalize_text(item)
            if not n or n in seen_norm:
                continue
            seen_norm.add(n)
            result.append(item)
    return result


def _tokens_set(phrases: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for phrase in phrases:
        for token in phrase_tokens(phrase):
            n = normalize_text(token)
            if n and is_informative_token(token):
                out.add(n)
    return out


def _non_weak_tokens(tokens: Iterable[str]) -> set[str]:
    return {t for t in tokens if t and not is_weak_context(t)}


def has_minimum_thematic_evidence(
    intent: "QueryIntent",
    matched_keywords: Iterable[str],
    article: "Article | None" = None,
) -> bool:
    """True when matched evidence includes at least one strong thematic signal.

    Strong signals:
    - original query phrase (informative)
    - informative core/equivalent phrase with a non-weak token
    - informative original/core/equiv single token that is not weak-only
    - multi-word concept match with a non-weak token tied to the query family
    - >=2 non-weak informative tokens from original/core/equiv/related
    """
    meaningful = filter_meaningful_matches(matched_keywords)
    if not meaningful:
        return False

    orig = intent.original_query.strip()
    core = [c for c in intent.core_concepts if c and c.strip()]
    equiv = [e for e in intent.equivalent_phrases if e and e.strip()]
    related = [r for r in intent.related_concepts if r and r.strip()]

    orig_norm = normalize_text(orig) if orig else ""
    core_norms = {normalize_text(c) for c in core}
    equiv_norms = {normalize_text(e) for e in equiv}

    orig_tokens = _tokens_set([orig]) if orig else set()
    core_tokens = _tokens_set(core)
    equiv_tokens = _tokens_set(equiv)
    related_tokens = _tokens_set(related)

    # Tokens that are standalone concepts in core/equiv (not only fragments
    # of multi-word phrases such as "busqueda" in "colectivos de busqueda").
    core_standalone = _tokens_set(c for c in core if len(c.split()) == 1)
    equiv_standalone = _tokens_set(e for e in equiv if len(e.split()) == 1)

    orig_non_weak = _non_weak_tokens(orig_tokens)
    core_non_weak = _non_weak_tokens(core_tokens)
    equiv_non_weak = _non_weak_tokens(equiv_tokens)
    core_standalone_non_weak = _non_weak_tokens(core_standalone)
    equiv_standalone_non_weak = _non_weak_tokens(equiv_standalone)

    matched_norms: set[str] = set()
    matched_token_norms: set[str] = set()

    for match in meaningful:
        matched_norms.add(normalize_text(match))
        for token in phrase_tokens(match):
            n = normalize_text(token)
            if n and is_informative_token(token):
                matched_token_norms.add(n)

    # Original query phrase.
    if orig_norm and orig_norm in matched_norms and phrase_has_informative_token(orig):
        return True

    # Core / equivalent phrases with a strong token.
    for match in meaningful:
        tokens = phrase_tokens(match)
        if len(tokens) < 2:
            continue
        n = normalize_text(match)
        if (n in core_norms or n in equiv_norms) and phrase_has_strong_token(match):
            return True

    # Informative non-weak single tokens from original / core / equivalent.
    single_token_evidence = {
        normalize_text(m) for m in meaningful if len(phrase_tokens(m)) == 1
    }
    multi_word_evidence_present = any(len(phrase_tokens(m)) > 1 for m in meaningful)
    for token_norm in single_token_evidence:
        if token_norm in orig_non_weak:
            return True
        # Standalone core/equiv concepts (inflacion, Banxico, IA, ...) count.
        if token_norm in core_standalone_non_weak or token_norm in equiv_standalone_non_weak:
            return True
        # Short distinctive acronyms from core/equiv may stand alone.
        if (token_norm in core_non_weak or token_norm in equiv_non_weak) and len(token_norm) <= 3:
            return True
        # Multi-word phrase fragments (busqueda/frente) need supporting evidence.
        if multi_word_evidence_present and (
            token_norm in core_non_weak or token_norm in equiv_non_weak
        ):
            return True

    # Multi-word concept match tied to original/core/equiv/related with a strong token.
    for match in meaningful:
        tokens = phrase_tokens(match)
        if len(tokens) < 2:
            continue
        phrase_token_norms = {
            normalize_text(t) for t in tokens if is_informative_token(t)
        }
        strong = {
            normalize_text(t)
            for t in tokens
            if is_informative_token(t) and not is_weak_context(t)
        }
        if not strong:
            continue
        if phrase_token_norms & (orig_tokens | core_tokens | equiv_tokens | related_tokens):
            return True
        n = normalize_text(match)
        if n in core_norms or n in equiv_norms or n == orig_norm:
            return True

    # Multiple non-weak informative tokens tied to the query family.
    support_tokens = matched_token_norms & (
        orig_tokens | core_tokens | equiv_tokens | related_tokens
    )
    if len(_non_weak_tokens(support_tokens)) >= 2:
        return True
    if len(_non_weak_tokens(matched_token_norms)) >= 2 and (
        _non_weak_tokens(matched_token_norms)
        & (orig_tokens | core_tokens | equiv_tokens | related_tokens)
    ):
        return True

    return False


def meaningful_match_score_proxy(matched_keywords: Iterable[str]) -> int:
    """Count of non-weak informative evidence items (for ranking diagnostics)."""
    meaningful = filter_meaningful_matches(matched_keywords)
    count = 0
    for match in meaningful:
        tokens = phrase_tokens(match)
        if len(tokens) == 1:
            if is_informative_token(tokens[0]) and not is_weak_context(tokens[0]):
                count += 1
        else:
            if phrase_has_strong_token(match):
                count += 1
    return count


@dataclass
class ConceptGroup:
    """Deterministic semantic component of a search intent.

    match_tokens may SOLO-satisfy the group (original component token or a
    single-token core/equivalent alternative). evidence_phrases satisfy the
    group only when the full phrase is matched — multi-word phrases never
    count via a single fragment token.
    """

    key: str
    label: str
    evidence_phrases: list[str]
    match_tokens: set[str]
    required_tokens: frozenset[str]
    tier: str
    is_original_component: bool


@dataclass
class CoverageResult:
    matched_labels: list[str]
    matched_original_labels: list[str]
    matched_related_labels: list[str]
    total_original: int
    total_related: int
    total_groups: int
    has_full_intent_phrase: bool
    full_phrase: str | None
    coverage_ratio: float
    passes: bool
    strongest_evidence: list[str]


def _dedupe_phrases(phrases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        if not p or not p.strip():
            continue
        n = normalize_text(p)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(p.strip())
    return out


def _non_weak_token_set(tokens: Iterable[str]) -> set[str]:
    return {normalize_text(t) for t in tokens if t and not is_weak_context(t)}


def _is_weak_label(label: str) -> bool:
    toks = phrase_tokens(label)
    if not toks:
        return False
    return all(is_weak_context(t) for t in toks)


def derive_concept_groups(intent: "QueryIntent") -> list[ConceptGroup]:
    """Build concept groups from existing QueryIntent only (no extra Gemini).

    Multi-token originals are decomposed into per-token components.
    Single-token core/equiv alternatives (inflacion, Banxico, IA) may solo-
    satisfy a component. Multi-word core/equiv/related phrases become full-
    phrase evidence only — fragments never solo-satisfy a component.
    """
    orig = (intent.original_query or "").strip()
    orig_raw = informative_phrase_tokens(orig)
    orig_norms = [normalize_text(t) for t in orig_raw]
    orig_norm_set = set(orig_norms)

    core_phrases = [c for c in intent.core_concepts if c and c.strip()]
    equiv_phrases = [e for e in intent.equivalent_phrases if e and e.strip()]
    related_phrases = [r for r in intent.related_concepts if r and r.strip()]

    # Single-concept / atomic intent.
    if len(orig_norms) <= 1:
        tokens: set[str] = set(orig_norms)
        phrases: list[str] = [orig] if orig else []
        key = orig_norms[0] if orig_norms else "query"
        label = orig or (orig_raw[0] if orig_raw else "query")
        for c in core_phrases + equiv_phrases:
            c_norms = [normalize_text(t) for t in informative_phrase_tokens(c)]
            if not c_norms:
                continue
            if normalize_text(c) == normalize_text(orig) or set(c_norms) & tokens:
                phrases.append(c)
                tokens |= set(c_norms)
            elif len(c_norms) == 1 and not is_weak_context(c):
                phrases.append(c)
                tokens.add(c_norms[0])
        for r in related_phrases:
            for t in informative_phrase_tokens(r):
                if not is_weak_context(t):
                    tokens.add(normalize_text(t))
        return [
            ConceptGroup(
                key=key,
                label=label,
                evidence_phrases=_dedupe_phrases(phrases),
                match_tokens=tokens,
                required_tokens=frozenset(),
                tier="original_atomic",
                is_original_component=True,
            )
        ]

    components: list[ConceptGroup] = []
    for raw, n in zip(orig_raw, orig_norms):
        # Solo-match tokens: original component token + single-token alts only.
        solo_tokens = {n}
        evidence = [raw]
        for c in core_phrases + equiv_phrases:
            c_norms = [normalize_text(t) for t in informative_phrase_tokens(c)]
            if not c_norms:
                continue
            shares = n in set(c_norms)
            if len(c_norms) == 1:
                cn = c_norms[0]
                if cn == n or cn in solo_tokens:
                    evidence.append(c)
                    continue
                if not is_weak_context(c):
                    # Strong single-token alternative (inflacion, Banxico, IA).
                    solo_tokens.add(cn)
                    evidence.append(c)
                elif shares:
                    evidence.append(c)
            elif shares:
                # Multi-word phrase containing this component: full-phrase evidence.
                evidence.append(c)
        components.append(
            ConceptGroup(
                key=n,
                label=raw,
                evidence_phrases=_dedupe_phrases(evidence),
                match_tokens=solo_tokens,
                required_tokens=frozenset(),
                tier="original_component",
                is_original_component=True,
            )
        )

    topic_idxs = [i for i, g in enumerate(components) if not _is_weak_label(g.label)]
    geo_idxs = [i for i, g in enumerate(components) if _is_weak_label(g.label)]
    if not topic_idxs:
        topic_idxs = list(range(len(components)))
    if not geo_idxs:
        geo_idxs = list(range(len(components)))

    def _attach_phrase(phrase: str, token: str, weak: bool) -> None:
        """Attach single-token alts as solo tokens; multi-word as phrase evidence."""
        idxs = geo_idxs if weak else topic_idxs
        for i in idxs:
            comps_tokens = components[i].match_tokens
            comps_evidence = components[i].evidence_phrases
            # Single-token alternatives may solo-satisfy.
            if len(phrase_tokens(phrase)) == 1 and not weak:
                comps_tokens.add(token)
            if phrase and phrase not in comps_evidence:
                comps_evidence.append(phrase)

    for c in core_phrases + equiv_phrases:
        c_norms = [normalize_text(t) for t in informative_phrase_tokens(c)]
        if len(c_norms) == 1:
            cn = c_norms[0]
            if cn in orig_norm_set:
                continue
            _attach_phrase(c, cn, is_weak_context(c))
        else:
            for t in c_norms:
                if t in orig_norm_set:
                    continue
                # Multi-word phrase tokens never solo-match; phrase evidence only.
                _attach_phrase(c, t, is_weak_context(t))

    groups = list(components)

    covered_tokens = set(orig_norm_set)
    for g in groups:
        covered_tokens |= g.match_tokens

    for r in related_phrases:
        r_norms = {normalize_text(t) for t in informative_phrase_tokens(r)}
        if not r_norms:
            continue
        if r_norms & orig_norm_set:
            continue
        non_weak = _non_weak_token_set(r_norms)
        if non_weak and all(t in covered_tokens for t in non_weak):
            continue
        # Related support is optional; multi-token phrases require phrase or
        # multi-token evidence at evaluation for stronger isolation.
        groups.append(
            ConceptGroup(
                key=normalize_text(r),
                label=r,
                evidence_phrases=[r],
                match_tokens=r_norms,
                required_tokens=frozenset(),
                tier="related_support",
                is_original_component=False,
            )
        )

    return groups


def evaluate_concept_coverage(
    intent: "QueryIntent",
    matched_keywords: Iterable[str],
    article: "Article | None" = None,
) -> CoverageResult:
    """Measure how much of the user's multi-part intent an article covers."""
    groups = derive_concept_groups(intent)
    meaningful = filter_meaningful_matches(matched_keywords)
    orig = (intent.original_query or "").strip()
    orig_norm = normalize_text(orig) if orig else ""

    core = [c for c in intent.core_concepts if c and c.strip()]
    equiv = [e for e in intent.equivalent_phrases if e and e.strip()]
    family_tokens: set[str] = set()
    for phrase in [orig, *core, *equiv]:
        for t in informative_phrase_tokens(phrase):
            family_tokens.add(normalize_text(t))

    article_match_norms = {normalize_text(m) for m in meaningful}
    article_tokens: set[str] = set()
    for m in meaningful:
        for t in phrase_tokens(m):
            if is_informative_token(t):
                article_tokens.add(normalize_text(t))

    has_full = False
    full_phrase: str | None = None
    if len(informative_phrase_tokens(orig)) > 1 and orig_norm and orig_norm in article_match_norms:
        has_full = True
        full_phrase = orig

    matched_original_labels: list[str] = []
    matched_related_labels: list[str] = []
    matched_labels: list[str] = []

    for g in groups:
        phrase_hit = any(
            normalize_text(p) in article_match_norms for p in g.evidence_phrases if p
        )
        solo_hit = bool(g.match_tokens & article_tokens)
        if g.required_tokens:
            group_ok = phrase_hit or g.required_tokens.issubset(article_tokens)
        elif g.tier == "related_support":
            non_weak_hits = _non_weak_token_set(g.match_tokens) & article_tokens
            if len(g.match_tokens) >= 2:
                reinforced = non_weak_hits & family_tokens
                group_ok = (
                    phrase_hit
                    or len(non_weak_hits) >= 2
                    or bool(reinforced)
                )
            else:
                group_ok = phrase_hit or len(non_weak_hits) >= 1
        else:
            group_ok = phrase_hit or solo_hit
        if not group_ok:
            continue
        matched_labels.append(g.label)
        if g.is_original_component:
            matched_original_labels.append(g.label)
        elif g.tier == "related_support":
            matched_related_labels.append(g.label)

    total_original = sum(1 for g in groups if g.is_original_component)
    total_related = sum(1 for g in groups if g.tier == "related_support")
    total_groups = len(groups)
    n_orig = len(matched_original_labels)
    n_rel = len(matched_related_labels)

    if has_full:
        passes = True
    elif total_original <= 1:
        passes = n_orig >= 1
    elif n_orig >= 2:
        passes = True
    elif n_orig == 1 and n_rel >= 1:
        # One original component plus independent related support.
        passes = True
    else:
        # Related-only or single ambiguous component is not enough.
        passes = False

    denom = max(total_original, 2) if total_original > 1 else max(total_original, 1)
    coverage_ratio = min(1.0, (n_orig + min(n_rel, 1)) / denom)

    strongest: list[str] = []
    if has_full and full_phrase:
        strongest.append(full_phrase)
    else:
        for label in matched_original_labels + matched_related_labels:
            if label not in strongest:
                strongest.append(label)
        for m in meaningful:
            if m not in strongest and len(phrase_tokens(m)) > 1:
                strongest.append(m)

    return CoverageResult(
        matched_labels=matched_labels,
        matched_original_labels=matched_original_labels,
        matched_related_labels=matched_related_labels,
        total_original=total_original,
        total_related=total_related,
        total_groups=total_groups,
        has_full_intent_phrase=has_full,
        full_phrase=full_phrase,
        coverage_ratio=coverage_ratio,
        passes=passes,
        strongest_evidence=strongest,
    )


def has_minimum_concept_coverage(
    intent: "QueryIntent",
    matched_keywords: Iterable[str],
    article: "Article | None" = None,
) -> bool:
    return evaluate_concept_coverage(intent, matched_keywords, article).passes


def concept_match_display(cov: CoverageResult) -> list[str]:
    """User-visible Match evidence at concept level (stopword-free)."""
    if cov.has_full_intent_phrase and cov.full_phrase:
        cleaned_full = filter_meaningful_matches([cov.full_phrase])
        return cleaned_full or [cov.full_phrase]
    labels = list(cov.matched_original_labels)
    for label in cov.matched_related_labels:
        if label not in labels:
            labels.append(label)
    if not labels:
        return []
    chosen = cov.matched_original_labels[:4] if len(cov.matched_original_labels) >= 2 else labels[:4]
    cleaned: list[str] = []
    for label in chosen:
        parts = filter_meaningful_matches([label])
        cleaned.extend(parts if parts else [label])
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in cleaned:
        n = normalize_text(item)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(item)
    return out[:4]
