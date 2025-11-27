"""
Entity extraction agent that uses simple heuristics to locate people,
organizations, dates, and locations within the document.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]


@dataclass
class Entity:
    name: str
    mentions: int
    context: Optional[str] = None
    role: Optional[str] = None
    type: Optional[str] = None


@dataclass
class EntitiesResult:
    people: list[Entity]
    organizations: list[Entity]
    dates: list[Entity]
    locations: list[Entity]


class EntityExtractorAgent:
    """
    Rough-and-ready entity extractor geared toward structured demo output.
    """

    MAX_INPUT_CHARS = 6000
    ENTITY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You extract structured entities from financial documents. Respond with JSON containing "
                    "people, organizations, dates, and locations arrays. Each entry needs name, mentions (int), "
                    "context (short quote), and role/type when applicable."
                ),
            ),
            ("human", "{document}"),
        ]
    )

    COMPANY_HINTS = ("Inc", "Corp", "LLC", "Ltd", "Company", "Bank", "Group")
    LOCATION_HINTS = (
        "New York",
        "London",
        "Mumbai",
        "Delhi",
        "Singapore",
        "San Francisco",
        "Tokyo",
    )

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client
        self._chain = self.ENTITY_PROMPT | llm_client if llm_client else None

    async def run(self, document_text: str) -> EntitiesResult:
        if self._chain and document_text.strip():
            try:
                return await self._extract_llm(document_text)
            except Exception:
                pass
        return await asyncio.to_thread(self._extract, document_text)

    async def _extract_llm(self, text: str) -> EntitiesResult:
        response = await self._chain.ainvoke({"document": text[: self.MAX_INPUT_CHARS]})
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)
        payload = json.loads(content)

        def parse_entities(key: str, fallback_type: Optional[str] = None) -> list[Entity]:
            items = payload.get(key) or []
            entities: list[Entity] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                entities.append(
                    Entity(
                        name=item.get("name", "").strip(),
                        mentions=int(item.get("mentions", 1) or 1),
                        context=item.get("context", ""),
                        role=item.get("role"),
                        type=item.get("type", fallback_type),
                    )
                )
            return [entity for entity in entities if entity.name]

        return EntitiesResult(
            people=parse_entities("people"),
            organizations=parse_entities("organizations", "company"),
            dates=parse_entities("dates", "date"),
            locations=parse_entities("locations", "location"),
        )

    def _extract(self, text: str) -> EntitiesResult:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        people = self._collect_people(text, sentences)
        orgs = self._collect_orgs(text, sentences)
        dates = self._collect_dates(text, sentences)
        locations = self._collect_locations(text, sentences)

        return EntitiesResult(people=people, organizations=orgs, dates=dates, locations=locations)

    def _collect_people(self, text: str, sentences: list[str]) -> list[Entity]:
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
        counts = Counter(matches)
        return self._build_entities(counts, sentences)

    def _collect_orgs(self, text: str, sentences: list[str]) -> list[Entity]:
        pattern = r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Bank|Group))\b"
        matches = re.findall(pattern, text)
        counts = Counter(matches)
        return self._build_entities(counts, sentences, entity_type="company")

    def _collect_dates(self, text: str, sentences: list[str]) -> list[Entity]:
        patterns = [
            r"\b20\d{2}\b",
            r"\b19\d{2}\b",
            r"\bQ[1-4]-?20\d{2}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b",
        ]
        matches: list[str] = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text))
        counts = Counter(matches)
        return self._build_entities(counts, sentences, entity_type="date")

    def _collect_locations(self, text: str, sentences: list[str]) -> list[Entity]:
        matches: list[str] = []
        for location in self.LOCATION_HINTS:
            if location in text:
                matches.append(location)
        counts = Counter(matches)
        return self._build_entities(counts, sentences, entity_type="location")

    def _build_entities(
        self,
        counts: Counter,
        sentences: list[str],
        entity_type: Optional[str] = None,
    ) -> list[Entity]:
        entities: list[Entity] = []
        for name, freq in counts.most_common(5):
            context = next(
                (sentence.strip() for sentence in sentences if name in sentence),
                "",
            )
            entities.append(
                Entity(
                    name=name,
                    mentions=freq,
                    context=context,
                    type=entity_type,
                )
            )
        return entities

