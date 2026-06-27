import torch

from torch import Tensor
from torch.nn import functional as F

from typing import Optional, Tuple, Union

from ..log.logger import get_logger

from ..exceptions.exception import (
    TensorShapeError,
    TensorTypeError,
    InterpolationValidationError,
    InterpolationError,
)

log = get_logger(__name__)


class TensorInterpolationService:
    """
    Utility service for safe tensor interpolation
    and image upscaling.
    """

    SUPPORTED_MODES = {
        "nearest",
        "bilinear",
        "bicubic",
        "area",
        "nearest-exact",
    }

    # ========================================================
    # Safe Interpolation
    # ========================================================

    @staticmethod
    def interpolate(
        x: Tensor,
        scale_factor: Optional[
            Union[int, float, Tuple[float, float]]
        ] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        mode: str = "bicubic",
        align_corners: Optional[bool] = False,
    ) -> Tensor:
        """
        Safely interpolate a tensor using
        PyTorch interpolation.
        """

        # ====================================================
        # Validation
        # ====================================================

        if not isinstance(x, torch.Tensor):

            raise TensorTypeError(
                "Input must be a torch.Tensor"
            )

        if x.dim() != 4:

            raise TensorShapeError(
                f"Expected 4D tensor [B,C,H,W], "
                f"got shape {tuple(x.shape)}"
            )

        if not torch.is_floating_point(x):

            raise TensorTypeError(
                f"Expected floating tensor, got dtype {x.dtype}"
            )

        if mode not in TensorInterpolationService.SUPPORTED_MODES:

            raise InterpolationValidationError(
                f"Unsupported mode: {mode}"
            )

        if scale_factor is None and size is None:

            raise InterpolationValidationError(
                "Either scale_factor or size must be provided"
            )

        # ====================================================
        # scale_factor validation
        # ====================================================

        if scale_factor is not None:

            if isinstance(scale_factor, (int, float)):

                if scale_factor <= 0:

                    raise InterpolationValidationError(
                        "scale_factor must be > 0"
                    )

            elif isinstance(scale_factor, tuple):

                if any(v <= 0 for v in scale_factor):

                    raise InterpolationValidationError(
                        "All scale_factor values must be > 0"
                    )

            else:

                raise InterpolationValidationError(
                    "Invalid scale_factor type"
                )

        # ====================================================
        # size validation
        # ====================================================

        if size is not None:

            if isinstance(size, int):

                if size <= 0:

                    raise InterpolationValidationError(
                        "size must be > 0"
                    )

            elif isinstance(size, tuple):

                if len(size) != 2:

                    raise InterpolationValidationError(
                        "size tuple must contain 2 values"
                    )

                if any(v <= 0 for v in size):

                    raise InterpolationValidationError(
                        "All size values must be > 0"
                    )

            else:

                raise InterpolationValidationError(
                    "Invalid size type"
                )

        log.debug(
            "interpolation_started | shape=%s | mode=%s",
            tuple(x.shape),
            mode,
        )

        # ====================================================
        # Build interpolation args
        # ====================================================

        kwargs = {
            "input": x,
            "mode": mode,
        }

        if size is not None:

            kwargs["size"] = size

        else:

            kwargs["scale_factor"] = scale_factor

        # align_corners supported only
        # for bilinear/bicubic
        if mode in {"bilinear", "bicubic"}:

            kwargs["align_corners"] = align_corners

        # ====================================================
        # Execute interpolation
        # ====================================================

        try:

            with torch.no_grad():

                out = F.interpolate(**kwargs)

                log.debug(
                    "interpolation_completed | output_shape=%s",
                    tuple(out.shape),
                )

                return out

        except RuntimeError as e:

            log.exception(
                "interpolation_failed | error=%s",
                str(e),
            )

            raise InterpolationError(
                f"Interpolation failed: {str(e)}"
            ) from e

    # ========================================================
    # Fallback Upscale
    # ========================================================

    @staticmethod
    def fallback_upscale(
        x: Tensor,
        scale_factor: int = 4,
        mode: str = "bicubic",
    ) -> Tensor:
        """
        Perform fallback tensor upscaling
        using interpolation.
        """

        log.warning(
            "fallback_upscale_triggered | mode=%s | scale=%d",
            mode,
            scale_factor,
        )

        return TensorInterpolationService.interpolate(
            x=x,
            scale_factor=scale_factor,
            mode=mode,
        )

    # ========================================================
    # Fast Preview Upscale
    # ========================================================

    @staticmethod
    def preview_upscale(
        x: Tensor,
        scale_factor: int = 4,
    ) -> Tensor:
        """
        Generate a fast low-cost preview upscale.
        """

        log.debug(
            "preview_upscale_started | scale=%d",
            scale_factor,
        )

        return TensorInterpolationService.interpolate(
            x=x,
            scale_factor=scale_factor,
            mode="nearest",
        )