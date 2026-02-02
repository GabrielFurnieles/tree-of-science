from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, Any

from .db.models import BatchStatus


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

    model_config = {"from_attributes": True}

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
            return datetime.fromtimestamp(v)
        return v
