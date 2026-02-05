from sqlalchemy import (
    Integer,
    String,
    Text,
    JSON,
    DateTime,
    ForeignKey,
    Enum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional, List
from datetime import datetime
from sqlalchemy import func
import enum

from .database import Base


# Enum definitions
class JobStatus(str, enum.Enum):
    """Job status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(str, enum.Enum):
    """
    Batch status values from OpenAI Batch API.

    Lifecycle:
    validating → validated -> in_progress → finalizing → completed
                                                       ↘ failed
                                                       ↘ expired
                                                       ↘ cancelled
    """

    VALIDATING = "validating"
    VALIDATED = "validated"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, index=True, autoincrement=True
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(
            JobStatus, native_enum=False, create_constraint=True, validate_strings=True
        ),
        insert_default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )
    model: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now("now", "utc")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now("now", "utc"),
        onupdate=func.now("now", "utc"),
    )
    finished_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    batches: Mapped[List["Batch"]] = relationship(
        "Batch", back_populates="job", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Job(id={self.id}, model='{self.model}', status='{self.status.value if self.status else None}')>"


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey(f"{Job.__tablename__}.id"), nullable=False, index=True
    )

    # OpenAI Batch API fields (matching JSONL structure)
    batch_oaid: Mapped[str] = mapped_column(String, index=True)  # OpenAI batch ID
    completion_window: Mapped[str]
    endpoint: Mapped[str]
    object: Mapped[str]
    status: Mapped[BatchStatus] = mapped_column(
        Enum(
            BatchStatus,
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
        ),
        nullable=False,
        default=BatchStatus.VALIDATING,
        index=True,
    )
    input_file_id: Mapped[str]  # OpenAI file ID
    local_file_id: Mapped[str]  # Local file path (from metadata.description)
    output_file_id: Mapped[Optional[str]]  # OpenAI output file ID
    error_file_id: Mapped[Optional[str]]  # OpenAI error file ID
    request_completed: Mapped[Optional[int]]  # flattened from request_counts object
    request_failed: Mapped[Optional[int]]  # flattened from request_counts object
    request_total: Mapped[Optional[int]]  # flattened from request_counts object
    usage: Mapped[dict] = mapped_column(
        JSON, nullable=True
    )  # e.g., {"prompt_tokens": 1000, "total_tokens": 1000}
    errors: Mapped[str] = mapped_column(Text, nullable=True)  # Error message if any
    created_at: Mapped[Optional[datetime]]
    cancelled_at: Mapped[Optional[datetime]]
    cancelling_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    expired_at: Mapped[Optional[datetime]]
    expires_at: Mapped[Optional[datetime]]
    failed_at: Mapped[Optional[datetime]]
    finalizing_at: Mapped[Optional[datetime]]
    in_progress_at: Mapped[Optional[datetime]]

    # Local tracking fields (optional)
    output_file_local: Mapped[Optional[str]]  # Local path where results are downloaded
    loaded_to_qdrant: Mapped[int] = mapped_column(Integer, default=0)  # Boolean: 0 or 1
    loaded_at: Mapped[Optional[datetime]]

    job: Mapped["Job"] = relationship(back_populates="batches")

    def __repr__(self):
        return f"<Batch(id={self.id}, batch_id='{self.batch_id}', status='{self.status.value if self.status else None}', local_file='{self.local_file_id}')>"
