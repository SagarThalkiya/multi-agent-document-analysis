"""
Lightweight summarizer agent. Falls back to heuristics when an LLM client
is not configured.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]


@dataclass
class SummaryResult:
    text: Optional[str]
    key_points: list[str]
    confidence: float


class SummarizerAgent:
    """
    Generate a concise summary and key points for the document.
    """

    MAX_INPUT_CHARS = 6000
    SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert financial analyst. Summarize the user's document in JSON with "
                    "keys summary_text (<=150 words), key_points (array of 3-5 concise bullet strings), "
                    "and confidence (0-1 float). Respond with JSON only."
                ),
            ),
            ("human", "{document}"),
        ]
    )

    def __init__(self, llm_client: Optional[Any] = None):
        self._llm_client = llm_client
        self._chain = self.SUMMARY_PROMPT | llm_client if llm_client else None

    async def run(self, document_text: str) -> SummaryResult:
        text = document_text.strip()
        if not text:
            return SummaryResult(text="", key_points=[], confidence=0.0)

        if self._chain:
            try:
                return await self._summarize_llm(text)
            except Exception as exc:
                import logging
                logger = logging.getLogger("multi-agent")
                logger.warning("Summarizer LLM call failed, falling back to heuristics: %s", exc)
                pass

        return await asyncio.to_thread(self._summarize, text)

    async def _summarize_llm(self, text: str) -> SummaryResult:
        response = await self._chain.ainvoke({"document": text[: self.MAX_INPUT_CHARS]})
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = " ".join(part["text"] for part in content if isinstance(part, dict) and "text" in part)
        payload = json.loads(content)
        summary_text = payload.get("summary_text") or payload.get("summary") or ""
        key_points = payload.get("key_points") or payload.get("keyPoints") or []
        confidence = float(payload.get("confidence", 0.85))

        key_points = [kp.strip() for kp in key_points if isinstance(kp, str) and kp.strip()][:5]
        words = summary_text.split()
        if len(words) > 150:
            summary_text = " ".join(words[:150]) + "..."

        return SummaryResult(text=summary_text.strip(), key_points=key_points, confidence=round(confidence, 2))

    def _summarize(self, text: str) -> SummaryResult:
        if not text:
            return SummaryResult(text="", key_points=[], confidence=0.0)

        sentences = re.split(r"(?<=[.!?])\s+", text)
        trimmed_sentences = [s.strip() for s in sentences if s.strip()]
        summary_text = " ".join(trimmed_sentences[:3])

        words = summary_text.split()
        if len(words) > 150:
            summary_text = " ".join(words[:150]) + "..."

        key_points = trimmed_sentences[:5]

        confidence = min(0.95, 0.5 + len(summary_text) / max(len(text), 1))

        return SummaryResult(
            text=summary_text,
            key_points=key_points,
            confidence=round(confidence, 2),
        )

