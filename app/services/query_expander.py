import json
import logging
import time

from app.config import EXPANSION_MODEL, GEMINI_TIMEOUT, MAX_EXPANDED_TERMS
from app.query_intent import QueryIntent

logger = logging.getLogger(__name__)

_EXPANSION_PROMPT = """Eres un asistente de busqueda de noticias en espanol. Analiza la consulta del usuario e identifica conceptos clave para buscar articulos relevantes en feeds RSS.

Consulta: "{query}"

Responde SOLO con un objeto JSON valido (sin markdown, sin explicaciones) con esta estructura:

{{
  "core": [
    "concepto original del usuario",
    "variaciones lexical cercanas al concepto original",
    "formulaciones alternativas fuertemente equivalentes"
  ],
  "equivalent": [
    "sinonimos y parafasis del mismo concepto",
    "formas como un periodista describiria lo mismo",
    "variaciones morfologicas y expresiones idiomaticas"
  ],
  "related": [
    "conceptos fuertemente relacionados pero NO sinonimos",
    "temas que aparecen en articulos sobre este sujet",
    "contexto que ayuda a encontrar cobertura relevante"
  ]
}}

Objetivo: generar aproximadamente 20-25 conceptos en total, distribuidos en los tres niveles.

Reglas estrictas:
- El primer elemento de "core" SIEMPRE debe ser la consulta original exacta del usuario
- Cada termino debe ser un concepto de significado completo y especifico
- Prefiere frases de varias palabras que describan conceptos concretos
- NO incluyas palabras genericas sueltas como: personas, Mexico, gobierno, pais, apoyo, politica, sistema, guerra, mercado, innovacion, tecnologia, datos, computacion, sociedad, autoridades
- NO incluyas abreviaturas cortas (2-3 letras) como terminos independientes
- Para abreviaturas como "IA", incluyelas SOLO como parte de una frase mas larga (ej: "modelos de IA", "sistemas de IA")
- "core" debe contener 4-5 conceptos de la interpretacion mas fuerte
- "equivalent" debe contener 7-8 sinonimos o parafasis genuinas
- "related" debe contener 8-12 conceptos fuertemente relacionados
- Si menos conceptos son genuinamente utiles, devuelve menos en vez de inventar asociaciones debiles
- Prioriza frases que un periodista usaria en un titulo de noticia
- Distingue claramente entre sinonimo/equivalente y meramente relacionado
- Genera alrededor de {max_terms} conceptos en total cuando sea posible"""


def expand_query(query: str) -> QueryIntent:
    from app.services.gemini import _get_client

    client = _get_client()
    if client is None:
        logger.info("Gemini unavailable, using original query for expansion")
        return QueryIntent(original_query=query, core_concepts=[query])

    prompt = _EXPANSION_PROMPT.format(query=query, max_terms=MAX_EXPANDED_TERMS)

    max_retries = 1
    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=EXPANSION_MODEL,
                contents=prompt,
            )
            raw = (response.text or "").strip()
            intent = _parse_expansion(raw, query)
            total = len(intent.all_phrases())
            logger.info(
                "Query expansion succeeded: %d terms for '%s' "
                "(core=%d, equiv=%d, related=%d)",
                total,
                query,
                len(intent.core_concepts),
                len(intent.equivalent_phrases),
                len(intent.related_concepts),
            )
            return intent
        except Exception as e:
            logger.warning(
                "Gemini expansion error (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                e,
            )
            if attempt < max_retries:
                time.sleep(2)

    logger.warning("Query expansion failed, falling back to original query")
    return QueryIntent(original_query=query, core_concepts=[query])


def _parse_expansion(raw: str, original_query: str) -> QueryIntent:
    if not raw:
        return QueryIntent(
            original_query=original_query, core_concepts=[original_query]
        )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse Gemini response as JSON, using fallback")
        return _parse_flat_fallback(cleaned, original_query)

    core = _clean_list(data.get("core", data.get("core_concepts", [])))
    equiv = _clean_list(data.get("equivalent", data.get("equivalent_phrases", [])))
    related = _clean_list(data.get("related", data.get("related_concepts", [])))

    all_terms = core + equiv + related
    if not all_terms:
        return QueryIntent(
            original_query=original_query, core_concepts=[original_query]
        )

    if original_query not in all_terms:
        core.insert(0, original_query)

    total = len(core) + len(equiv) + len(related)
    if total > MAX_EXPANDED_TERMS:
        max_related = max(8, MAX_EXPANDED_TERMS // 3)
        max_equiv = max(7, MAX_EXPANDED_TERMS // 3)
        max_core = MAX_EXPANDED_TERMS - max_equiv - max_related
        core = core[:max_core]
        equiv = equiv[:max_equiv]
        related = related[:max_related]

    return QueryIntent(
        original_query=original_query,
        core_concepts=core,
        equivalent_phrases=equiv,
        related_concepts=related,
    )


def _parse_flat_fallback(raw: str, original_query: str) -> QueryIntent:
    terms = [t.strip().strip('"').strip("'") for t in raw.split(",")]
    terms = [t for t in terms if t]
    if not terms:
        return QueryIntent(
            original_query=original_query, core_concepts=[original_query]
        )
    if original_query not in terms:
        terms.insert(0, original_query)
    terms = terms[:MAX_EXPANDED_TERMS]
    return QueryIntent(
        original_query=original_query,
        core_concepts=terms,
        equivalent_phrases=[],
        related_concepts=[],
    )


def _clean_list(items: list) -> list[str]:
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            cleaned = item.strip().strip('"').strip("'")
            if cleaned:
                result.append(cleaned)
    return result
