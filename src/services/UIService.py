from src.exceptions.exception import ModelValidationError

from src.services.inferenceService import upscale_full
import gradio as gr


# Main Gradio Function
def gradio_interface(
    image,
    tile_mode,
    log,
    tile_service,
    runner,
    device,
    progress=gr.Progress(),
):
    try:

        log.info("request_received | tile_mode=%s", tile_mode)

        if image is None:
            raise ModelValidationError("No image uploaded")

        # Tile Mode
        if tile_mode:

            log.info("tile_mode_selected")

            upscaled_image = tile_service.upscale(
                image=image,
                progress=progress,
            )

        # Full Mode
        else:

            log.info("full_mode_selected")

            upscaled_image = upscale_full(
                image=image,
                runner=runner,
                device=device,
            )

        log.info(
            "request_completed | original_size=%s | upscaled_size=%s",
            image.size,
            upscaled_image.size,
        )

        return (
            image,
            upscaled_image,
            f"Original: {image.size[0]}x{image.size[1]}",
            f"Upscaled: {upscaled_image.size[0]}x{upscaled_image.size[1]}",
        )

    except Exception as e:

        log.exception("request_failed | error=%s", str(e))

        return (None, None, "Error", str(e))
