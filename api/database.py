from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, DateTime, select, delete
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import os
from datetime import timedelta

POSTGRES_URL = os.getenv(
    "POSTGRES_URL", 
    "postgresql+asyncpg://appuser:changeme_in_production@postgres:5432/disaster_damage"
)

engine = create_async_engine(POSTGRES_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class InferenceJob(Base):
    __tablename__ = "inference_jobs"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    status: Mapped[str] = mapped_column(String, index=True)
    pre_image_path: Mapped[str] = mapped_column(String)
    post_image_path: Mapped[str] = mapped_column(String)
    disaster_name: Mapped[str] = mapped_column(String)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    processing_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    total_buildings: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    no_damage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    minor_damage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    major_damage: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    destroyed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    geojson_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    map_html_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    client_ip: Mapped[Optional[str]] = mapped_column(String, nullable=True)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def create_job(db: AsyncSession, job_data: dict) -> InferenceJob:
    job = InferenceJob(**job_data)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def get_job(db: AsyncSession, job_id: str) -> Optional[InferenceJob]:
    result = await db.execute(select(InferenceJob).filter(InferenceJob.id == job_id))
    return result.scalars().first()


async def update_job(db: AsyncSession, job_id: str, **kwargs) -> Optional[InferenceJob]:
    job = await get_job(db, job_id)
    if not job:
        return None
    for key, value in kwargs.items():
        setattr(job, key, value)
    await db.commit()
    await db.refresh(job)
    return job


async def list_recent_jobs(db: AsyncSession, limit: int = 20) -> List[InferenceJob]:
    result = await db.execute(select(InferenceJob).order_by(InferenceJob.created_at.desc()).limit(limit))
    return result.scalars().all()


async def delete_old_jobs(db: AsyncSession, older_than_hours: int = 48) -> int:
    cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
    result = await db.execute(delete(InferenceJob).where(InferenceJob.created_at < cutoff))
    await db.commit()
    return result.rowcount
