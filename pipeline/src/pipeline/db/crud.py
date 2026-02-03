from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from rich.logging import RichHandler
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

        db.expunge(job)  # persist job obj after session
        return job

    @staticmethod
    def get_job(db: Session, job_id: int):
        stmt = select(models.Job).where(models.Job.id == job_id)
        return db.execute(stmt)

    @staticmethod
    def update_job_status(db: Session, job_id: int):
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

        job = JobRepository.get_job(job_id)
        if new_status is not None and new_status != job.status:
            job.status = new_status
            db.commit()
            db.refresh(job)

        return job


class BatchRepository:

    @staticmethod
    def create_batch(db: Session, batch: schemas.BatchCreate):
        batch = models.Batch(**batch.model_dump())
        db.add(batch)
        db.commit()
        db.refresh(batch)

        db.expunge(batch)
        return batch

    @staticmethod
    def get_batches(db: Session, job_id: int):
        stmt = select(models.Batch).where(models.Batch.job_id == job_id)
        return db.scalars(stmt).all()

    @staticmethod
    def update_batch(db: Session, batch_id: int, update_data: schemas.BatchUpdate):
        batch = db.execute(select(models.Batch).where(models.Batch.id == batch_id))
        update_dict = update_data.model_dump(exclude_unset=True)

        for k, v in update_dict.items():
            setattr(batch, k, v)

        db.commit()
        db.refresh(batch)
        return batch
