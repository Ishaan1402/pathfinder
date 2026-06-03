import os
import contextlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from schema import Base

# Read database URL from environment or fallback to local SQLite database.
# Note: For SQLite, we want to enable check_same_thread=False to support multiple threads/connections.
DATABASE_URL = os.getenv("HPO_DATABASE_URL", "sqlite:///hpo_studies.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database and creates all tables."""
    Base.metadata.create_all(bind=engine)

@contextlib.contextmanager
def get_db_session():
    """Context manager for DB sessions to ensure automatic closing and transactional safety."""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
