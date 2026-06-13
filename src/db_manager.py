import os
import contextlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .schema import Base

# Read database URL from environment or fallback to local SQLite database.
# Note: For SQLite, we want to enable check_same_thread=False to support multiple threads/connections.
DATABASE_URL = os.getenv("HPO_DATABASE_URL", "sqlite:///hpo_studies.db")

from sqlalchemy import event

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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes the database, creates tables, and runs additive migrations."""
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

    # Migrate data from the legacy segmentation_metrics table to trial_results if present.
    if "segmentation_metrics" in existing_tables:
        _migrate_segmentation_metrics_to_trial_results()


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
    "study_reviews": {
        "estimated_score_improvement": "FLOAT",
        "cited_best_trial": "INTEGER",
        "confidence": "VARCHAR(50) DEFAULT 'high'",
        "baseline_best_score": "FLOAT",
        "applied_at_completed_count": "INTEGER",
        "applied_at": "DATETIME",
        "actual_score_improvement": "FLOAT",
        "outcome_measured_at": "DATETIME",
        "outcome_status": "VARCHAR(30) DEFAULT 'pending'",
        "quality_flagged": "BOOLEAN DEFAULT 0",
    },
    "agent_reasoning_logs": {
        "estimated_score_improvement": "FLOAT",
        "actual_score_improvement": "FLOAT",
    },
    "study_status": {
        "health_tier": "VARCHAR(50) DEFAULT 'healthy'",
        "health_reason": "TEXT",
        "health_updated_at": "DATETIME",
        "nudge_dismissed_trials": "INTEGER",
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
                # Check for old column name to copy data
                old_name = None
                if col_name == "estimated_score_improvement" and "estimated_dice_improvement" in present:
                    old_name = "estimated_dice_improvement"
                elif col_name == "actual_score_improvement" and "actual_dice_improvement" in present:
                    old_name = "actual_dice_improvement"

                with engine.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}'))
                    if old_name:
                        conn.execute(text(f'UPDATE {table} SET {col_name} = {old_name}'))
            except Exception as e:
                # Best-effort: a concurrent process may have added it already.
                print(f"Error migrating column {col_name} in {table}: {e}")

    # Drop obsolete columns after confirming new columns are present and data is copied
    for table, col_map in [
        ("agent_reasoning_logs", {"estimated_dice_improvement": "estimated_score_improvement", "actual_dice_improvement": "actual_score_improvement"}),
        ("study_reviews", {"estimated_dice_improvement": "estimated_score_improvement", "actual_dice_improvement": "actual_score_improvement"}),
    ]:
        if table in existing_tables:
            try:
                present = {c["name"] for c in inspector.get_columns(table)}
            except Exception:
                continue
            for old_col, new_col in col_map.items():
                if old_col in present:
                    try:
                        with engine.begin() as conn:
                            # If new column exists, migrate remaining NULL values if any
                            if new_col in present:
                                conn.execute(text(f"UPDATE {table} SET {new_col} = {old_col} WHERE {new_col} IS NULL"))
                            # Drop the obsolete column
                            conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {old_col}"))
                            print(f"Migration: Dropped obsolete column '{old_col}' from table '{table}'")
                    except Exception as drop_err:
                        print(f"Migration: Error dropping obsolete column '{old_col}' from table '{table}': {drop_err}")


def _migrate_segmentation_metrics_to_trial_results():
    from sqlalchemy import inspect, text
    try:
        with engine.begin() as conn:
            # Check if trial_results table exists and is empty
            res = conn.execute(text("SELECT COUNT(*) FROM trial_results")).fetchone()
            if res and res[0] > 0:
                return

            study_name = os.getenv("HPO_STUDY_NAME", "seg_v1")
            try:
                s_res = conn.execute(text("SELECT study_name FROM study_status LIMIT 1")).fetchone()
                if s_res:
                    study_name = s_res[0]
                else:
                    s_res = conn.execute(text("SELECT study_name FROM study_reviews LIMIT 1")).fetchone()
                    if s_res:
                        study_name = s_res[0]
            except Exception:
                pass

            inspector = inspect(engine)
            seg_cols = {c["name"] for c in inspector.get_columns("segmentation_metrics")}

            cols_to_select = []
            cols_to_insert = []

            mapping = {
                "trial_id": "trial_id",
                "epoch_reached": "epoch_reached",
                "final_dice_score": "primary_score",
                "final_bce_loss": "primary_loss",
                "val_loss_history": "score_history_json",
                "weights_path": "weights_path",
                "gpu_model": "gpu_model",
                "max_vram_gb": "max_vram_gb",
                "oom_triggered": "oom_triggered",
                "created_at": "created_at"
            }

            for old_col, new_col in mapping.items():
                if old_col in seg_cols:
                    cols_to_select.append(old_col)
                    cols_to_insert.append(new_col)

            if cols_to_select:
                select_clause = ", ".join(cols_to_select)
                insert_clause = ", ".join(cols_to_insert)

                stmt = f"""
                    INSERT INTO trial_results (study_name, {insert_clause})
                    SELECT :study_name, {select_clause} FROM segmentation_metrics
                """
                conn.execute(text(stmt), {"study_name": study_name})
                print(f"Successfully migrated data from segmentation_metrics to trial_results.")
    except Exception as e:
        print(f"Error migrating segmentation_metrics data to trial_results: {e}")


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
