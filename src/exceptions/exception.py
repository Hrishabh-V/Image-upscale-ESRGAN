from ..log import get_logger

log = get_logger(__name__)


class ModelError(Exception):
    """
    Base exception for all model-related errors.
    """

    log_message = "Model error"

    def __init__(self, message: str):
        log.error("%s: %s", self.log_message, message)
        super().__init__(message)


# Validation Exceptions
class ModelValidationError(ModelError):
    """
    Base validation exception.
    """

    log_message = "Validation error"


class TensorShapeError(ModelValidationError):
    """
    Raised when a tensor has an invalid or unexpected shape.
    """

    log_message = "Tensor shape error"

class UnsupportedScaleError(ModelValidationError):
    """
    Raised when an unsupported image scaling factor is requested.
    """

    log_message = "Unsupported scale error"

    

class TensorTypeError(ModelValidationError):
    """
    Raised when a tensor has an invalid or unsupported dtype.
    """

    log_message = "Tensor type error"


class TensorNaNError(ModelValidationError):
    """
    Raised when a tensor contains NaN or invalid numerical values.
    """

    log_message = "Tensor NaN error"


class InterpolationValidationError(ModelValidationError):
    """
    Raised when interpolation parameters or inputs are invalid.
    """

    log_message = "Interpolation validation error"


# Runtime Exceptions
class ModelRuntimeError(ModelError):
    """
    Base runtime exception for model execution failures.
    """

    log_message = "Runtime error"


class ModelForwardError(ModelRuntimeError):
    """
    Raised when model forward execution fails.
    """

    log_message = "Model forward error"


class InterpolationError(ModelRuntimeError):
    """
    Raised when interpolation fails during runtime.
    """

    log_message = "Interpolation runtime error"


class ResidualBlockError(ModelRuntimeError):
    """
    Raised when a residual block execution fails.
    """

    log_message = "Residual block error"


class UpsampleError(ModelRuntimeError):
    """
    Raised when upsampling operation fails.
    """

    log_message = "Upsample error"
