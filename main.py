# ESRGAN Upscaler (Full + Tile Mode)
import logging
from src.log.logger import configure_logging, get_logger
from src.UI.initialiser import create_and_launch_ui, initialize_services

configure_logging(level=logging.DEBUG)
log = get_logger(__name__)


def main():

    log.info("app_launch_started")

    tile_service, runner, device = initialize_services()

    create_and_launch_ui(tile_service, runner, device)


# Run
if __name__ == "__main__":

    main()
