"""
Shared Pydantic schemas for the FastAPI application.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: Literal["uploaded"]
    message: str


class AnalyzeRequest(BaseModel):
    job_id: str = Field(..., description="Job identifier returned by /upload")


class AnalyzeResponse(BaseModel):
    job_id: str
    status: Literal["processing"]
    message: str


class SummaryPayload(BaseModel):
    text: Optional[str] = None
    key_points: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


class EntityItem(BaseModel):
    name: str
    mentions: int
    context: Optional[str] = None
    role: Optional[str] = None
    type: Optional[str] = None


class EntitiesPayload(BaseModel):
    people: list[EntityItem] = Field(default_factory=list)
    organizations: list[EntityItem] = Field(default_factory=list)
    dates: list[EntityItem] = Field(default_factory=list)
    locations: list[EntityItem] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


class SentimentPayload(BaseModel):
    tone: Optional[str] = None
    confidence: Optional[float] = None
    formality: Optional[str] = None
    key_phrases: list[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    error: Optional[str] = None


class AnalysisResults(BaseModel):
    summary: SummaryPayload | dict = Field(default_factory=SummaryPayload)
    entities: EntitiesPayload | dict = Field(default_factory=EntitiesPayload)
    sentiment: SentimentPayload | dict = Field(default_factory=SentimentPayload)


class ResultsResponse(BaseModel):
    job_id: str
    status: Literal["uploaded", "processing", "completed", "partial", "failed"]
    document_name: str
    results: AnalysisResults
    total_processing_time_seconds: Optional[float] = None
    agents_completed: int = 0
    agents_failed: int = 0
    warning: Optional[str] = None

