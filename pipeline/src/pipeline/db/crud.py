from sqlalchemy import Engine, select, func
from sqlalchemy.orm import Session
from rich.logging import RichHandler
from datetime import datetime
import logging

from . import database
from . import models
from .. import schemas

logging.basicConfig(
    level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger()


class DBRepository:

    @staticmethod
    def create_db(engine: Engine):
        return database.Base.metadata.create_all(bind=engine)

    @staticmethod
    def delete_db(engine: Engine):
        return database.Base.metadata.drop_all(bind=engine)

    @staticmethod
    def recreate_db(engine: Engine):
        DBRepository.delete_db(engine)
        return DBRepository.create_db(engine)


class JobRepository:

    @staticmethod
    def create_job(db: Session, model: str):
        job = models.Job(model=model)
        db.add(job)
        db.commit()
        db.refresh(job)

        return schemas.JobRead.model_validate(job)

    @staticmethod
    def get_job(db: Session, job_id: int):
        stmt = select(models.Job).where(models.Job.id == job_id)
        job = db.execute(stmt).scalar_one()

        return schemas.JobRead.model_validate(job)

    @staticmethod
    def update_job_status(db: Session, job_id: int) -> models.Job | None:
        batches = BatchRepository.get_batches(db, job_id)
        statuses = {b.status for b in batches}

        if not batches:
            logger.warning(
                f"No batches found for job '{job_id}'. Skipping job status update."
            )
            return None

        new_status = None

        if statuses & {models.BatchStatus.CANCELLED, models.BatchStatus.FAILED}:
            new_status = models.JobStatus.FAILED
        elif statuses & {models.BatchStatus.IN_PROGRESS, models.BatchStatus.FINALIZING}:
            new_status = models.JobStatus.PROCESSING
        elif all([s == models.JobStatus.COMPLETED for s in statuses]):
            new_status = models.JobStatus.COMPLETED

        job = db.execute(select(models.Job).where(models.Job.id == job_id)).scalar_one()

        if new_status is not None and new_status != job.status:
            job.status = new_status
            db.commit()
            db.refresh(job)

        return schemas.JobRead.model_validate(job)


class BatchRepository:

    @staticmethod
    def create_batch(db: Session, batch: schemas.BatchCreate):
        batch = models.Batch(**batch.model_dump())
        db.add(batch)
        db.commit()
        db.refresh(batch)

        return schemas.BatchRead.model_validate(batch)

    @staticmethod
    def get_batches(db: Session, job_id: int):
        stmt = select(models.Batch).where(models.Batch.job_id == job_id)
        batches = db.execute(stmt).scalars().all()
        return [schemas.BatchRead.model_validate(b) for b in batches]

    @staticmethod
    def update_batch(db: Session, id: int, update_data: schemas.BatchUpdate):
        timestamp_cols = [
            models.Batch.created_at,
            models.Batch.cancelled_at,
            models.Batch.cancelling_at,
            models.Batch.completed_at,
            models.Batch.expired_at,
            models.Batch.failed_at,
            models.Batch.finalizing_at,
            models.Batch.in_progress_at,
        ]

        calculated_update_at = func.max(
            *[func.coalesce(col, datetime.min) for col in timestamp_cols]
        )

        result = db.execute(
            select(
                models.Batch,
                calculated_update_at.label("calculated_update_at"),
            ).where(models.Batch.id == id)
        ).one()

        batch, calc_date = result

        # Update batch with new data
        update_dict = update_data.model_dump(exclude_unset=True)

        for k, v in update_dict.items():
            setattr(batch, k, v)

        db.commit()
        db.refresh(batch)

        # Add calculated field for display
        batch.calculated_update_at = calc_date

        return schemas.BatchRead.model_validate(batch)
