from functools import partial
from src.log.logger import get_logger
from src.core.config import MODEL_PATH
from src.utils.util import check_cuda
from src.UI.interface import create_interface
from src.services.UIService import gradio_interface
from src.architecture.RRDBnet import RRDBRunner
from src.services.tileservice import TileInferenceService
from src.exceptions.exception import ModelRuntimeError


log = get_logger(__name__)


def initialize_services() -> TileInferenceService:
    """
    Initialize model and tile inference service
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    device = check_cuda()

    try:

        log.info("model_loading_started | " "device=%s | path=%s", device, MODEL_PATH)

        runner = RRDBRunner(MODEL_PATH, device=device)

        log.info("model_loading_completed")

    except (RuntimeError, FileNotFoundError, OSError) as e:

        raise ModelRuntimeError(f"Failed to initialize ESRGAN model: {str(e)}") from e

    tile_service = TileInferenceService(runner=runner, scale=4, tile_size=128)

    log.info("tile_service_initialized | " "scale=%d | tile_size=%d", 4, 128)

    return tile_service, runner, device


def create_and_launch_ui(
    tile_service: TileInferenceService,
    runner: RRDBRunner,
    device: str,
) -> None:
    """
    Build and launch Gradio UI
    """

    interface = create_interface(
        gradio_interface=partial(
            gradio_interface,
            log=log,
            tile_service=tile_service,
            runner=runner,
            device=device,
        )
    )

    interface.launch()
