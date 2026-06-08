# Test package init — sets up shared test database and sys.path
import os
import sys
import tempfile

# Add project root to sys.path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Create a shared temporary database for all test modules.
# Since each test uses unique study names, there is no conflict.
# This must be set BEFORE any src.* imports to ensure db_manager
# resolves DATABASE_URL from this env var.
_test_db_fd, TEST_DB_PATH = tempfile.mkstemp(suffix=".db", prefix="hpo_test_suite_")
os.close(_test_db_fd)
os.environ["HPO_DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

import atexit

def _cleanup_test_db():
    for suffix in ("", "-shm", "-wal"):
        path = TEST_DB_PATH + suffix
        try:
            os.unlink(path)
        except OSError:
            pass

atexit.register(_cleanup_test_db)
