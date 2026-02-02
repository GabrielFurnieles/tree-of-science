from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from contextlib import contextmanager
from decouple import config

# Engine configuration
engine = create_engine(config("SQLITE_URI"))

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Base class for ORM models
class Base(DeclarativeBase):
    pass


@contextmanager
def get_session():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise e
    finally:
        db.close()
