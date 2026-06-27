import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..exceptions.handler import  handle_forward_error
from ..log import get_logger
from src.utils.util import make_layer

from ..exceptions.exception import (
    TensorShapeError,
    ResidualBlockError,
    ModelForwardError,
    UpsampleError,
    InterpolationError,
    InterpolationValidationError,
    ModelRuntimeError,
    TensorTypeError,
)
log = get_logger(__name__)




"""
Residual Dense Block (5 conv version)
"""
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        try:
            # Shape validation
            if x.dim() != 4:
                raise TensorShapeError(
                    f"Expected 4D tensor [B,C,H,W], got shape {x.shape}"
                )

            if x.size(1) != self.conv1.in_channels:
                raise TensorShapeError(
                    f"Expected {self.conv1.in_channels} input channels, "
                    f"got {x.size(1)}"
                )

            
            x1 = self.lrelu(self.conv1(x))

            # Concat validation
            if x.shape[2:] != x1.shape[2:]:
                raise TensorShapeError(
                    "Spatial dimensions mismatch between x and x1"
                )

            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

            out = x5 * 0.2 + x

            
            return out

        # Re-raise known custom exceptions
        except (
            TensorShapeError
        ):
            raise

        # Catch unexpected runtime failures
        except Exception as e:
            raise ResidualBlockError(
                f"ResidualDenseBlock_5C failed: {str(e)}"
            ) from e


"""RRDB Block"""
class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        try:
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
            return out * 0.2 + x
        except Exception as e:
            handle_forward_error("RRDB", e)


""" RRDBNet (ESRGAN Original Compatible)
"""
class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()

        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)

        # IMPORTANT: must match pretrained ESRGAN
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)

        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        try:
            # Input shape validation
            if x.dim() != 4:
                raise TensorShapeError(
                    f"Expected 4D tensor [B,C,H,W], got shape {x.shape}"
                )

            if x.size(1) != self.conv_first.in_channels:
                raise TensorShapeError(
                    f"Expected {self.conv_first.in_channels} input channels, "
                    f"got {x.size(1)}"
                )

            fea = self.conv_first(x)

            trunk = self.trunk_conv(self.RRDB_trunk(fea))
            fea = fea + trunk

            # Interpolation validation
            if fea.dim() != 4:
                raise InterpolationValidationError(
                    f"Interpolation expects 4D tensor, got shape {fea.shape}"
                )

            if fea.shape[-1] <= 0 or fea.shape[-2] <= 0:
                raise InterpolationValidationError(
                    f"Invalid spatial dimensions for interpolation: {fea.shape}"
                )

            # Upsample stage 1
            try:
                fea = F.interpolate(
                    fea,
                    scale_factor=2,
                    mode="nearest"
                )
            except Exception as e:
                raise InterpolationError(
                    f"Interpolation stage 1 failed: {str(e)}"
                ) from e

            try:
                fea = self.lrelu(self.upconv1(fea))
            except Exception as e:
                raise UpsampleError(
                    f"Upsample stage 1 failed: {str(e)}"
                ) from e

            # Interpolation validation
            if fea.dim() != 4:
                raise InterpolationValidationError(
                    f"Interpolation expects 4D tensor, got shape {fea.shape}"
                )

            if fea.shape[-1] <= 0 or fea.shape[-2] <= 0:
                raise InterpolationValidationError(
                    f"Invalid spatial dimensions for interpolation: {fea.shape}"
                )

            # Upsample stage 2
            try:
                fea = F.interpolate(
                    fea,
                    scale_factor=2,
                    mode="nearest"
                )
            except Exception as e:
                raise InterpolationError(
                    f"Interpolation stage 2 failed: {str(e)}"
                ) from e

            try:
                fea = self.lrelu(self.upconv2(fea))
            except Exception as e:
                raise UpsampleError(
                    f"Upsample stage 2 failed: {str(e)}"
                ) from e

            out = self.conv_last(self.lrelu(self.HRconv(fea)))
            return out

        except (
            TensorShapeError,
            InterpolationValidationError,
            InterpolationError,
            UpsampleError
        ):
            raise

        except Exception as e:
            raise ModelForwardError(
                f"RRDBNet forward failed: {str(e)}"
            ) from e

"""
Runner
"""
class RRDBRunner:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda"

        log.info("Loading RRDBNet on %s", self.device)

        self.model = RRDBNet(3, 3)

        try:
            state = torch.load(model_path, map_location=self.device)

            # support both raw and EMA checkpoints
            if "params_ema" in state:
                state = state["params_ema"]

            self.model.load_state_dict(state, strict=True)

        except Exception as e:
            raise ModelRuntimeError(
                f"Failed to load RRDB model weights: {str(e)}"
            ) from e

        self.model.to(self.device)
        self.model.eval()

        if self.use_cuda:
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to(memory_format=torch.channels_last)

            try:
                self.model = torch.compile(self.model)
                log.info("Torch compile enabled")
            except Exception as e:
                log.warning("Torch compile failed: %s", str(e))     

    @torch.no_grad()
    def infer(self, x):
        try:
            # Tensor shape validation
            if x.dim() != 4:
                raise TensorShapeError(
                    f"Expected 4D tensor [B,C,H,W], got shape {x.shape}"
                )

            # Tensor dtype validation
            if not torch.is_floating_point(x):
                raise TensorTypeError(
                    f"Expected floating tensor, got dtype {x.dtype}"
                )

            # Channel validation
            if x.size(1) != 3:
                raise TensorShapeError(
                    f"Expected 3 input channels, got {x.size(1)}"
                )

            x = x.to(self.device, non_blocking=self.use_cuda)

            if self.use_cuda:
                x = x.to(memory_format=torch.channels_last)

                with torch.amp.autocast("cuda"):
                    out = self.model(x)
            else:
                out = self.model(x)

            return out

        except (
            TensorShapeError,
            TensorTypeError
        ):
            raise

        except Exception as e:
            raise ModelRuntimeError(
                f"RRDBRunner inference failed: {str(e)}"
            ) from e
