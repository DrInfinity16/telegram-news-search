from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.evidence import clean_token, informative_tokens_from_phrases, is_informative_token

logger = logging.getLogger(__name__)

MIN_SINGLE_TOKEN_CHARS = 4
AMBIGUOUS_TOKEN_MAX_CHARS = 3


@dataclass
class QueryIntent:
    original_query: str
    core_concepts: list[str] = field(default_factory=list)
    equivalent_phrases: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)

    def all_phrases(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for term in [self.original_query, *self.core_concepts, *self.equivalent_phrases, *self.related_concepts]:
            if term not in seen:
                seen.add(term)
                result.append(term)
        return result

    def multi_word_phrases(self) -> list[str]:
        return [p for p in self.all_phrases() if len(p.split()) > 1]

    def meaningful_single_tokens(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for phrase in self.all_phrases():
            words = phrase.split()
            if len(words) == 1 and len(words[0]) >= MIN_SINGLE_TOKEN_CHARS:
                if not is_informative_token(words[0]):
                    continue
                if words[0] not in seen:
                    seen.add(words[0])
                    result.append(words[0])
        return result

    def retrieval_tokens(self) -> list[str]:
        """Informative single tokens for lexical candidate retrieval.

        Includes tokens from multi-word phrases so queries like
        "guerra Rusia Ucrania" can generate candidates when words appear
        non-contiguously. Stopwords are excluded. Short tokens remain
        eligible when informative and must use boundary-aware matching.
        """
        return informative_tokens_from_phrases(self.all_phrases())

    def search_terms(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for term in self.all_phrases():
            if term not in seen:
                seen.add(term)
                result.append(term)
            words = term.split()
            if len(words) > 1:
                for word in words:
                    token = clean_token(word)
                    if (
                        token
                        and token not in seen
                        and len(token) >= MIN_SINGLE_TOKEN_CHARS
                        and is_informative_token(token)
                    ):
                        seen.add(token)
                        result.append(token)
        return result

    def tier_phrases(self) -> tuple[list[str], list[str], list[str]]:
        multi = lambda terms: [t for t in terms if len(t.split()) > 1]
        return (
            [self.original_query],
            multi(self.core_concepts) + multi(self.equivalent_phrases),
            multi(self.related_concepts),
        )

    def generic_single_tokens(self) -> list[str]:
        tokens: list[str] = []
        for phrase in self.all_phrases():
            words = phrase.split()
            if len(words) == 1 and len(words[0]) < MIN_SINGLE_TOKEN_CHARS:
                if is_informative_token(words[0]):
                    tokens.append(words[0])
        return tokens

    def token_is_ambiguous(self, token: str) -> bool:
        return len(token) <= AMBIGUOUS_TOKEN_MAX_CHARS
