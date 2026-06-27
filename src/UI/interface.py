import gradio as gr


def create_interface(gradio_interface):
    """
    Create Gradio interface
    """

    return gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Image(type="pil"),
            gr.Checkbox(label=("Enable Tile Mode " "(recommended for large images)")),
        ],
        outputs=[
            gr.Image(label="Original"),
            gr.Image(label="Upscaled"),
            gr.Text(label="Original Size"),
            gr.Text(label="Upscaled Size"),
        ],
        title=("ESRGAN Upscaler " "(Full + Tile Mode)"),
        description=(
            "ESRGAN image upscaler "
            "with optional tile inference "
            "for large images."
        ),
    )
