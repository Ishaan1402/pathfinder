
import contextlib
import logging
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from .schema import Base
from .settings import settings

logger = logging.getLogger(__name__)

DATABASE_URL = settings.database_url

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    connect_args["timeout"] = 30

engine = create_engine(DATABASE_URL, connect_args=connect_args)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Initializes the database, creates tables, and runs additive migrations."""
    # Ensure parent directory exists for the default SQLite path (.data/)
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.isdir(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    from sqlalchemy import inspect

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        existing_tables = set()

    # Drop legacy study_status table if it contains the obsolete coordinator_pending column
    if "study_status" in existing_tables:
        try:
            from sqlalchemy import text
            cols = {c["name"] for c in inspector.get_columns("study_status")}
            if "coordinator_pending" in cols:
                with engine.begin() as conn:
                    conn.execute(text("DROP TABLE study_status"))
                existing_tables.remove("study_status")
        except Exception as e:
            print(f"Migration: Error dropping legacy study_status table: {e}")
            

    # Create all defined tables in schema
    Base.metadata.create_all(bind=engine)

    # Run additive migrations for altered tables
    _apply_additive_migrations()


# Additive, idempotent column migrations for tables that predate a new field.
# Keeps an already-populated SQLite DB from 500ing when the ORM adds a nullable column.
_ADDITIVE_COLUMNS = {
    "trial_results": {
        "worker_id": "VARCHAR(100)",
        "git_commit": "VARCHAR(40)",
        "dataset_version": "VARCHAR(200)",
        "health_tier": "VARCHAR(50)",
        "health_reason": "TEXT",
    },

    "study_status": {
        "health_tier": "VARCHAR(50) DEFAULT 'healthy'",
        "health_reason": "TEXT",
        "health_updated_at": "DATETIME",
    }
}


def _apply_additive_migrations():
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
    except Exception:
        return

    for table, columns in _ADDITIVE_COLUMNS.items():
        if table not in existing_tables:
            continue  # create_all already built it with the full schema
        try:
            present = {c["name"] for c in inspector.get_columns(table)}
        except Exception:
            continue
        for col_name, col_type in columns.items():
            if col_name in present:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}'))
            except Exception as e:
                # Best-effort: a concurrent process may have added it already.
                print(f"Error migrating column {col_name} in {table}: {e}")




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

def get_or_create_study_status(session, study_name: str):
    from src.schema import StudyStatus
    status = session.query(StudyStatus).filter_by(study_name=study_name).first()
    if not status:
        status = StudyStatus(study_name=study_name)
        session.add(status)
    return status
