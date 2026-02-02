from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from . import database
from . import models
from .. import schemas


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

        db.expunge(job)
        return job

    @staticmethod
    def get_job(db: Session, job_id: int):
        stmt = select(models.Job).where(models.Job.id == job_id)
        return db.execute(stmt)


class BatchRepository:

    @staticmethod
    def create_batch(db: Session, batch: schemas.BatchCreate):
        batch = models.Batch(**batch.model_dump())
        db.add(batch)
        db.commit()
        db.refresh(batch)

        return batch
