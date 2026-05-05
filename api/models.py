"""
api/models.py
──────────────────────────────────────────────────────────────────────────────
FastAPI Pydantic v2 models for request/response validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details for Error Reporting"""
    type: str
    title: str
    status: int
    detail: str
    instance: str
    
    model_config = ConfigDict(populate_by_name=True)


class InferenceRequest(BaseModel):
    disaster_name: str = "Unknown Disaster"
    map_threshold: float = Field(default=0.35, ge=0.1, le=0.9)


class DamageStats(BaseModel):
    total_buildings: int
    no_damage: int
    minor_damage: int
    major_damage: int
    destroyed: int


class InferenceStatus(BaseModel):
    task_id: str
    status: Literal["PENDING", "PROGRESS", "SUCCESS", "FAILURE"]
    step: Optional[str] = None
    progress_pct: Optional[int] = None
    created_at: datetime


class InferenceResult(BaseModel):
    task_id: str
    status: str
    map_url: str
    geojson_url: str
    stats: DamageStats
    processing_time_seconds: float
