import pytest
from app.config import settings

@pytest.fixture(autouse=True)
def disable_nli_guard_for_tests():
    """Automatically disable NLI guard during unit test run to keep tests fast and offline."""
    orig_val = settings.enable_nli_guard
    settings.enable_nli_guard = False
    yield
    settings.enable_nli_guard = orig_val
