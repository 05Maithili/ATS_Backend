"""
Database configuration and models.
Uses SQLite for development, can be switched to PostgreSQL for production.
"""
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, String, DateTime, Text, Float, Integer, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    # Database - Use SQLite for development
    DATABASE_URL: str = "sqlite:///./ats_database.db"
    
    # For production, use PostgreSQL:
    # DATABASE_URL: str = "postgresql://user:password@localhost/ats_db"
    
    # JWT Settings
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenRouter Settings
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "openai/gpt-3.5-turbo"
    OPENROUTER_REFERER: str = "http://localhost"
    OPENROUTER_APP_NAME: str = "ai-ats-optimizer"
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields


settings = Settings()

# Create engine based on database type
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite needs check_same_thread=False for async
    engine = create_engine(
        settings.DATABASE_URL.replace("sqlite:///", "sqlite:///"),
        connect_args={"check_same_thread": False}
    )
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace("sqlite:///", "sqlite+aiosqlite:///"),
        connect_args={"check_same_thread": False}
    )
else:
    # PostgreSQL
    engine = create_engine(settings.DATABASE_URL)
    async_engine = create_async_engine(settings.DATABASE_URL)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# ============== Database Models ==============

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    resumes: Mapped[list["Resume"]] = relationship("Resume", back_populates="user", cascade="all, delete-orphan")
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")


class Resume(Base):
    """Resume model for storing user resumes."""
    __tablename__ = "resumes"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_content: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)  # Extracted text
    skills: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)  # JSON string
    experience_years: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="resumes")
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="resume", cascade="all, delete-orphan")


class JobDescription(Base):
    """Job Description model."""
    __tablename__ = "job_descriptions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    company: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Text] = mapped_column(Text, nullable=False)
    requirements: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses: Mapped[list["Analysis"]] = relationship("Analysis", back_populates="job_description")


class Analysis(Base):
    """Analysis result model."""
    __tablename__ = "analyses"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    resume_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("resumes.id"), nullable=True)
    job_description_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("job_descriptions.id"), nullable=True)
    
    # Score fields
    ats_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    skill_overlap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    semantic_alignment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    formatting_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Detailed results (stored as JSON)
    gaps: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)  # JSON string
    matches: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)  # JSON string
    recommendations: Mapped[Optional[Text]] = mapped_column(Text, nullable=True)  # JSON string
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")
    resume: Mapped[Optional["Resume"]] = relationship("Resume", back_populates="analyses")
    job_description: Mapped[Optional["JobDescription"]] = relationship("JobDescription", back_populates="analyses")


# ============== Database Functions ==============

async def get_db():
    """Get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def init_db_sync():
    """Initialize database tables (sync version for startup)."""
    Base.metadata.create_all(bind=engine)
