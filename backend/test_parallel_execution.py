"""
Tests that verify the three agents execute in parallel (not sequentially).
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from app.agents.entity_extractor import EntityExtractorAgent, EntitiesResult
from app.agents.sentiment import SentimentAgent, SentimentResult
from app.agents.summarizer import SummarizerAgent, SummaryResult
from app.services.orchestrator import AnalysisOrchestrator


@pytest.mark.asyncio
async def test_parallel_execution_with_mock_delays():
    """Ensure total wall time ~= slowest agent, not sum."""

    async def delayed_summary(_: str):
        await asyncio.sleep(0.4)
        return SummaryResult(text="summary", key_points=[], confidence=0.8)

    async def delayed_entities(_: str):
        await asyncio.sleep(0.2)
        return EntitiesResult(people=[], organizations=[], dates=[], locations=[])

    async def delayed_sentiment(_: str):
        await asyncio.sleep(0.3)
        return SentimentResult(tone="neutral", confidence=0.7, formality="formal", key_phrases=[])

    summarizer = MagicMock()
    summarizer.run = delayed_summary
    entity = MagicMock()
    entity.run = delayed_entities
    sentiment = MagicMock()
    sentiment.run = delayed_sentiment

    orchestrator = AnalysisOrchestrator(summarizer, entity, sentiment)
    start = time.perf_counter()
    _, outcomes = await orchestrator.run("doc")
    total = time.perf_counter() - start

    assert total < 0.6  # should be close to 0.4 (slowest)
    assert outcomes["summary"].success
    assert outcomes["entities"].success
    assert outcomes["sentiment"].success


@pytest.mark.asyncio
async def test_parallel_execution_real_agents():
    summarizer = SummarizerAgent()
    entity = EntityExtractorAgent()
    sentiment = SentimentAgent()
    orchestrator = AnalysisOrchestrator(summarizer, entity, sentiment)

    text = "John Smith led Acme Corp in 2024. Profits were strong." * 200
    start = time.perf_counter()
    _, outcomes = await orchestrator.run(text)
    total = time.perf_counter() - start

    max_agent_time = max(
        outcomes["summary"].processing_time_seconds,
        outcomes["entities"].processing_time_seconds,
        outcomes["sentiment"].processing_time_seconds,
    )
    assert total <= max_agent_time + 0.1

