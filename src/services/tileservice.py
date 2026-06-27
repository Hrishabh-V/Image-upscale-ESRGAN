# Safe Tile Inference Service

import numpy as np
import torch

from PIL import Image

from src.log.logger import get_logger

from src.utils.util import (
    numpy_to_pil,
    pil_to_numpy,
    image_to_tensor,
    tensor_to_image,
    clear_cuda,
    validate_scale,
    validate_tensor,
)

from src.services.interPolationService import TensorInterpolationService

from src.exceptions.exception import (
    TensorShapeError,
    TensorTypeError,
    ModelRuntimeError,
    InterpolationError,
)

log = get_logger(__name__)


class TileInferenceService:
    """Service for memory-efficient tiled image super-resolution inference."""

    def __init__(self, runner, scale=4, tile_size=128, tile_pad=16):
        """Initialize tile inference configuration."""

        self.runner = runner
        self.scale = scale
        self.tile_size = tile_size
        self.tile_pad = tile_pad

    # Tile Upscale
    def upscale(self, image, progress=None):
        """Upscale an image using tiled inference with optional fallback interpolation."""

        log.info(
            "tile_inference_started | " "scale=%d | tile_size=%d | " "tile_pad=%d",
            self.scale,
            self.tile_size,
            self.tile_pad,
        )

        if not isinstance(image, Image.Image):

            raise TensorTypeError(
                f"Expected PIL.Image.Image, got {type(image).__name__}"
            )

        # RGB Safety
        image = image.convert("RGB")

        # PIL -> NumPy
        img = pil_to_numpy(image)
        validate_scale(self.scale)

        h, w, _ = img.shape

        if h <= 0 or w <= 0:

            raise TensorShapeError(f"Invalid image dimensions: {(h, w)}")

        # Output Buffer
        out = np.zeros((h * self.scale, w * self.scale, 3), dtype=np.float32)

        # Tile Grid
        tiles_x = (w + self.tile_size - 1) // self.tile_size

        tiles_y = (h + self.tile_size - 1) // self.tile_size

        total_tiles = tiles_x * tiles_y

        log.info(
            "tile_grid_created | "
            "width=%d | height=%d | "
            "tiles_x=%d | tiles_y=%d | "
            "total_tiles=%d",
            w,
            h,
            tiles_x,
            tiles_y,
            total_tiles,
        )

        tile_counter = 0

        # Process Tiles
        for y in range(tiles_y):

            for x in range(tiles_x):

                tile_counter += 1

                if progress:

                    progress(
                        tile_counter / total_tiles,
                        desc=(
                            f"Processing Tile "
                            f"{tile_counter}/{total_tiles}"
                        ),
                    )

                # Original Tile Coordinates
                x1 = x * self.tile_size
                y1 = y * self.tile_size

                x2 = min(x1 + self.tile_size, w)

                y2 = min(y1 + self.tile_size, h)

                # Padded Tile Coordinates
                px1 = max(x1 - self.tile_pad, 0)

                py1 = max(y1 - self.tile_pad, 0)

                px2 = min(x2 + self.tile_pad, w)

                py2 = min(y2 + self.tile_pad, h)

                log.debug(
                    "tile_processing_started | "
                    "tile=%d/%d | "
                    "x1=%d | y1=%d | "
                    "x2=%d | y2=%d | "
                    "px1=%d | py1=%d | "
                    "px2=%d | py2=%d",
                    tile_counter,
                    total_tiles,
                    x1,
                    y1,
                    x2,
                    y2,
                    px1,
                    py1,
                    px2,
                    py2,
                )

                tensor = None
                sr_tensor = None
                sr = None

                try:

                    # Extract Padded Tile

                    tile = img[py1:py2, px1:px2]

                    # Tile -> Tensor
                    tensor = image_to_tensor(tile).to(self.runner.device)

                    validate_tensor(tensor, "tile_tensor")

                    # ESRGAN Inference

                    try:

                        with torch.no_grad():

                            sr_tensor = self.runner.infer(tensor)

                    # Fallback
                    except (RuntimeError, ModelRuntimeError) as e:

                        log.warning(
                            "tile_esrgan_failed | " "tile=%d/%d | " "error=%s",
                            tile_counter,
                            total_tiles,
                            str(e),
                        )

                        try:

                            sr_tensor = TensorInterpolationService.fallback_upscale(
                                tensor, scale_factor=self.scale, mode="bicubic"
                            )

                        except Exception as fallback_error:

                            raise InterpolationError(
                                f"Fallback upscale failed: " f"{str(fallback_error)}"
                            ) from fallback_error

                    # Tensor -> Image
                    sr = tensor_to_image(sr_tensor)

                    # Crop Padding
                    crop_x1 = (x1 - px1) * self.scale

                    crop_y1 = (y1 - py1) * self.scale

                    crop_x2 = crop_x1 + (x2 - x1) * self.scale

                    crop_y2 = crop_y1 + (y2 - y1) * self.scale

                    sr = sr[crop_y1:crop_y2, crop_x1:crop_x2]

                    # Validate Shape
                    expected_h = (y2 - y1) * self.scale

                    expected_w = (x2 - x1) * self.scale

                    if sr.shape[:2] != (expected_h, expected_w):

                        raise TensorShapeError(
                            "Output shape mismatch | "
                            f"expected="
                            f"{(expected_h, expected_w)} | "
                            f"got={sr.shape[:2]}"
                        )

                    # Output Placement
                    ox1 = x1 * self.scale
                    oy1 = y1 * self.scale

                    ox2 = ox1 + expected_w
                    oy2 = oy1 + expected_h

                    out[oy1:oy2, ox1:ox2] = sr

                    log.debug(
                        "tile_processing_completed | " "tile=%d/%d",
                        tile_counter,
                        total_tiles,
                    )

                except (
                    TensorShapeError,
                    TensorTypeError,
                    ModelRuntimeError,
                    InterpolationError,
                ):
                    raise

                except Exception as e:

                    log.exception(
                        "tile_processing_failed | " "tile=%d/%d | " "error=%s",
                        tile_counter,
                        total_tiles,
                        str(e),
                    )

                    raise

                finally:

                    # Cleanup
                    if tensor is not None:
                        del tensor

                    if sr_tensor is not None:
                        del sr_tensor

                    if sr is not None:
                        del sr

                    clear_cuda()

        # Final Conversion
        out = np.clip(out * 255, 0, 255).astype(np.uint8)

        log.info("tile_inference_completed | " "total_tiles=%d", total_tiles)

        return numpy_to_pil(out)
