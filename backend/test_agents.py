"""
Comprehensive unit tests for all three agents: Summarizer, EntityExtractor, Sentiment.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.entity_extractor import EntitiesResult, EntityExtractorAgent
from app.agents.sentiment import SentimentAgent, SentimentResult
from app.agents.summarizer import SummarizerAgent, SummaryResult


class TestSummarizerAgent:
    """Test suite for SummarizerAgent."""

    @pytest.mark.asyncio
    async def test_empty_text(self):
        agent = SummarizerAgent()
        result = await agent.run("")
        assert result.text == ""
        assert result.key_points == []
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_heuristic_summary(self):
        agent = SummarizerAgent()
        text = (
            "Sentence one. Sentence two has more detail. "
            "Sentence three continues. Sentence four ends."
        )
        result = await agent.run(text)
        assert isinstance(result, SummaryResult)
        assert len(result.key_points) <= 5
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_llm_summary(self):
        mock_llm = MagicMock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "summary_text": "LLM summary",
                        "key_points": ["a", "b", "c"],
                        "confidence": 0.9,
                    }
                )
            )
        )
        agent = SummarizerAgent(llm_client=mock_llm)
        agent._chain = mock_chain
        result = await agent.run("Doc text")
        assert result.text == "LLM summary"
        assert len(result.key_points) == 3
        assert result.confidence == 0.9


class TestEntityExtractorAgent:
    """Test suite for EntityExtractorAgent."""

    @pytest.mark.asyncio
    async def test_heuristic_people(self):
        agent = EntityExtractorAgent()
        text = "John Smith met Jane Doe in New York."
        result = await agent.run(text)
        assert isinstance(result, EntitiesResult)
        assert result.people

    @pytest.mark.asyncio
    async def test_llm_entities(self):
        mock_llm = MagicMock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "people": [{"name": "Alice", "mentions": 2}],
                        "organizations": [{"name": "Acme Corp", "mentions": 1}],
                        "dates": [],
                        "locations": [],
                    }
                )
            )
        )
        agent = EntityExtractorAgent(llm_client=mock_llm)
        agent._chain = mock_chain
        result = await agent.run("Doc")
        assert result.people[0].name == "Alice"
        assert result.organizations[0].name == "Acme Corp"


class TestSentimentAgent:
    """Test suite for SentimentAgent."""

    @pytest.mark.asyncio
    async def test_heuristic_sentiment(self):
        agent = SentimentAgent()
        text = "We saw strong growth and profits."
        result = await agent.run(text)
        assert result.tone in {"positive", "negative", "neutral"}
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_llm_sentiment(self):
        mock_llm = MagicMock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "tone": "positive",
                        "confidence": 0.95,
                        "formality": "formal",
                        "key_phrases": ["great results", "strong outlook"],
                    }
                )
            )
        )
        agent = SentimentAgent(llm_client=mock_llm)
        agent._chain = mock_chain
        result = await agent.run("Doc")
        assert result.tone == "positive"
        assert result.confidence == 0.95

