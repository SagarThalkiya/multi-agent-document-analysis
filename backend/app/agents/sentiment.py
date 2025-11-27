"""
Sentiment analysis agent using a lightweight lexicon-based approach.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]


POSITIVE_WORDS = {
    "growth",
    "improved",
    "record",
    "strong",
    "positive",
    "bullish",
    "expansion",
    "profit",
    "increase",
    "exceeded",
    "resilient",
}

NEGATIVE_WORDS = {
    "decline",
    "drop",
    "loss",
    "negative",
    "weak",
    "missed",
    "risk",
    "volatility",
    "uncertain",
    "downturn",
    "slowdown",
}


@dataclass
class SentimentResult:
    tone: str
    confidence: float
    formality: str
    key_phrases: list[str]


class SentimentAgent:
    """
    Determine tone, confidence, formality, and supportive phrases.
    """

    MAX_INPUT_CHARS = 6000
    SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You evaluate financial documents. Respond strictly as JSON with keys tone "
                    "(positive/negative/neutral), confidence (0-1), formality (formal/informal), and "
                    "key_phrases (array of 2-3 short quotes supporting the tone)."
                ),
            ),
            ("human", "{document}"),
        ]
    )

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client
        self._chain = self.SENTIMENT_PROMPT | llm_client if llm_client else None

    async def run(self, document_text: str) -> SentimentResult:
        if self._chain and document_text.strip():
            try:
                return await self._analyze_llm(document_text)
            except Exception:
                pass
        return await asyncio.to_thread(self._analyze, document_text)

    async def _analyze_llm(self, text: str) -> SentimentResult:
        response = await self._chain.ainvoke({"document": text[: self.MAX_INPUT_CHARS]})
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)
        payload = json.loads(content)
        tone = payload.get("tone", "neutral")
        confidence = float(payload.get("confidence", 0.85))
        formality = payload.get("formality", "formal")
        key_phrases = payload.get("key_phrases") or payload.get("keyPhrases") or []
        key_phrases = [kp.strip() for kp in key_phrases if isinstance(kp, str) and kp.strip()][:3]
        return SentimentResult(
            tone=tone,
            confidence=round(confidence, 2),
            formality=formality,
            key_phrases=key_phrases,
        )

    def _analyze(self, text: str) -> SentimentResult:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        positive_hits = sum(1 for token in tokens if token in POSITIVE_WORDS)
        negative_hits = sum(1 for token in tokens if token in NEGATIVE_WORDS)

        if positive_hits > negative_hits:
            tone = "positive"
        elif negative_hits > positive_hits:
            tone = "negative"
        else:
            tone = "neutral"

        total_hits = positive_hits + negative_hits
        confidence = min(0.95, 0.5 + total_hits / max(len(tokens), 1) * 10)
        formality = "informal" if any("'" in word for word in tokens) else "formal"

        sentences = re.split(r"(?<=[.!?])\s+", text)
        key_phrases = []
        for sentence in sentences:
            lowered = sentence.lower()
            if (
                (tone == "positive" and any(pos in lowered for pos in POSITIVE_WORDS))
                or (tone == "negative" and any(neg in lowered for neg in NEGATIVE_WORDS))
                or tone == "neutral"
            ):
                key_phrases.append(sentence.strip())
            if len(key_phrases) >= 3:
                break

        return SentimentResult(
            tone=tone,
            confidence=round(confidence, 2),
            formality=formality,
            key_phrases=key_phrases,
        )

