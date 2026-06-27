import logging

from src.utils.util import (
    convert_to_rgb,
    pil_to_numpy,
    numpy_to_tensor,
    tensor_to_numpy,
    numpy_to_pil
    
    
    )
import torch
import numpy as np
from PIL import Image


from src.log.logger import (
    configure_logging,
    get_logger
)
configure_logging(level=logging.DEBUG)
log = get_logger(__name__)

def upscale_full(image,runner,device):

    try:

        log.info(
            "full_inference_started"
        )

        
        # RGB Safety
        image = convert_to_rgb(image)

        # PIL -> NumPy
        img = pil_to_numpy(image)

        # NumPy -> Tensor
        tensor = numpy_to_tensor(img, device=device)

        # ESRGAN Inference
        with torch.no_grad():

            sr = runner.infer(tensor)

        # Tensor -> NumPy
        sr = tensor_to_numpy(sr)

        # NumPy -> PIL
        out = numpy_to_pil(sr)

        log.info(
            "full_inference_completed"
        )

        return out

    except Exception as e:

        log.exception(
            "full_inference_failed | error=%s",
            str(e)
        )

        raise
