from ..log import get_logger
from .exception import (
    ModelForwardError
)

log = get_logger(__name__)

# ---------------- Generic Wrappers ----------------


def handle_forward_error(module: str, error: Exception):
    """Handle model forward-pass execution failures."""
    log.error("[%s] Forward error: %s", module, str(error))
    raise ModelForwardError(f"{module}: {error}") from error