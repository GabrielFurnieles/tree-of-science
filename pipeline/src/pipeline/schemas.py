from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from .db.models import JobStatus, BatchStatus


class JobRead(BaseModel):
    id: int
    status: JobStatus
    model: str

    model_config = ConfigDict(from_attributes=True)

    def __repr__(self):
        return f"<Job(id={self.id}, model='{self.model}', status='{self.status.value if self.status else None}')>"


class BatchCreate(BaseModel):
    # FK
    job_id: int

    # OpenAI Batch API fields
    batch_id: str
    completion_window: str
    endpoint: str
    object: str
    status: BatchStatus = Field(default=BatchStatus.VALIDATING)

    input_file_id: str
    local_file_id: str
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None

    # Request counts
    request_completed: Optional[int] = None
    request_failed: Optional[int] = None
    request_total: Optional[int] = None

    usage: Optional[Dict[str, Any]] = None
    errors: Optional[str] = None

    # Timestamps
    created_at: datetime
    cancelled_at: Optional[datetime] = None
    cancelling_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    finalizing_at: Optional[datetime] = None
    in_progress_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator(
        "created_at",
        "cancelled_at",
        "cancelling_at",
        "completed_at",
        "expired_at",
        "expires_at",
        "failed_at",
        "finalizing_at",
        "in_progress_at",
        mode="before",
    )
    @classmethod
    def unix_to_datetime(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v


class BatchUpdate(BaseModel):
    status: Optional[BatchStatus] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    request_completed: Optional[int] = None
    request_failed: Optional[int] = None
    request_total: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    errors: Optional[str] = None
    cancelled_at: Optional[datetime] = None
    cancelling_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    finalizing_at: Optional[datetime] = None
    in_progress_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("*", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v


class BatchRead(BaseModel):
    id: int
    job_id: int
    batch_id: str
    status: BatchStatus
    input_file_id: str
    local_file_id: str
    output_file_id: Optional[str] = None
    request_completed: Optional[int] = None
    request_failed: Optional[int] = None
    request_total: Optional[int] = None
    usage: Optional[Dict[str, Any]] = None
    errors: Optional[str] = None
    created_at: datetime
    calculated_update_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("*", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    def __repr__(self):
        return f"<Batch(id={self.id}, batch_id='{self.batch_id}', status='{self.status.value if self.status else None}', local_file='{self.local_file_id}'), created_at={self.created_at}>"
