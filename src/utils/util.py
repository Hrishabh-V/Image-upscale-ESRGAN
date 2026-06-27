import numpy as np
import torch
from PIL import Image

from ..log import get_logger
from ..exceptions import (
    TensorShapeError,
    TensorTypeError,
    TensorNaNError,
    ModelRuntimeError,
    UnsupportedScaleError
)

from ..core.config import VALIDATE_TENSOR

log = get_logger(__name__)

VALID_SCALE = 4


# Tensor Validation
def validate_tensor(
    x: torch.Tensor,
    name: str = "tensor",
) -> None:
    """
    Validate tensor format.

    Expected:
        Shape : [B, C, H, W]
        Type  : floating point
        Values: finite
    """

    if not isinstance(x, torch.Tensor):
        raise TensorTypeError(
            f"{name} must be torch.Tensor"
        )

    if x.ndim != 4:
        raise TensorShapeError(
            f"{name} must be 4D [B,C,H,W], "
            f"got shape {tuple(x.shape)}"
        )

    if not x.is_floating_point():
        raise TensorTypeError(
            f"{name} must be floating point"
        )

    # Faster NaN + Inf validation
    # Expensive validation only in DEBUG mode
    if VALIDATE_TENSOR == "DEBUG":

        if not torch.isfinite(x).all():
            raise TensorNaNError(
                f"NaN or Inf detected in {name}"
            )

    log.debug(
        "Validated tensor '%s' | shape=%s dtype=%s device=%s",
        name,
        tuple(x.shape),
        x.dtype,
        x.device,
    )


# Scale Validation
def validate_scale(scale: int) -> None:
    """
    Validate upscale factor.
    Supports only 4x.
    """

    if scale != VALID_SCALE:
        raise UnsupportedScaleError(
            f"Only {VALID_SCALE}x upscaling is supported "
            f"(got scale={scale})"
        )


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB.
    """
    if not isinstance(image, Image.Image):
        raise TensorTypeError(
            f"Expected PIL.Image.Image, got {type(image).__name__}"
        )
    return image.convert("RGB")


# PIL -> NumPy
def pil_to_numpy(
    image: Image.Image,
) -> np.ndarray:
    """
    Convert PIL image -> NumPy float32 array.
    Output range: [0, 1]
    """

    if not isinstance(image, Image.Image):
        raise TensorTypeError(
            f"Expected PIL.Image.Image, got {type(image).__name__}"
        )

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


# NumPy -> PIL
def numpy_to_pil(
    array: np.ndarray,
) -> Image.Image:
    """
    Convert NumPy image -> PIL image.
    Accepts:
        float32 [0,1]
        uint8   [0,255]
    """

    if not isinstance(array, np.ndarray):
        raise TensorTypeError(
            f"Expected np.ndarray, got {type(array).__name__}"
        )

    if array.dtype != np.uint8:
        array = (
            array * 255.0
        ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(array)

# Image -> Tensor
def image_to_tensor(
    image: np.ndarray,
) -> torch.Tensor:
    """
    Convert image HWC -> BCHW tensor.
    """

    if not isinstance(image, np.ndarray):
        raise TensorTypeError(
            f"Expected np.ndarray, got {type(image).__name__}"
        )

    if image.ndim != 3:
        raise TensorShapeError(
            f"Expected HWC image, got shape {image.shape}"
        )

    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
    )


# Tensor -> Image
def tensor_to_image(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert tensor BCHW -> image HWC.
    """

    validate_tensor(
        tensor,
        name="tensor_to_image_input"
    )

    return (
        tensor.squeeze(0)
        .detach()
        .clamp_(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def tensor_to_numpy(sr):
    """
    Convert a PyTorch tensor image to a NumPy image array.
    """

    sr = (
        sr.squeeze()
        .float()
        .cpu()
        .clamp(0, 1)
        .numpy()
    )

    sr = np.transpose(sr, (1, 2, 0))

    return sr
# numpy to tensor
def numpy_to_tensor(
    img: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Convert a NumPy image array (H, W, C) to a PyTorch tensor
    with shape (1, C, H, W).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(
            f"Expected np.ndarray, got {type(img).__name__}"
        )

    tensor = (
        torch.from_numpy(
            np.transpose(img, (2, 0, 1))
        )
        .float()
        .unsqueeze(0)
        .to(device)
    )

    return tensor




"""
Creates a stack of repeated neural network blocks.

Args:
    block: Neural network block class/function.
    n_layers (int): Number of blocks.

Returns:
    nn.Sequential: Stacked layers.
"""
def make_layer(block, n_layers):

    if not isinstance(n_layers, int):
        raise TensorTypeError(
            f"n_layers must be int, got {type(n_layers).__name__}"
        )

    if n_layers <= 0:
        raise TensorShapeError(
            "n_layers must be > 0"
        )

    layers = []

    for _ in range(n_layers):
        layers.append(block())

    return torch.nn.Sequential(*layers)





# CUDA Cleanup
def clear_cuda() -> None:
    """
    Clear unused CUDA memory.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""utilise gpu"""
def check_cuda():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    log.info(
        "device_initialized | device=%s",
        device
    )

    return device




def load_model(model_class, model_path: str, device: str):
    try:
        log.info("model_loading_started")

        model = model_class(
            model_path,
            device=device
        )

        log.info("model_loading_completed")

        return model

    except Exception as e:
        log.exception("model_loading_failed")

        raise ModelRuntimeError(
            f"Failed to initialize model: {e}"
        ) from e