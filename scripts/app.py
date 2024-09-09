import gradio as gr
from PIL import Image
import torch
import numpy as np
import sys
import os
import RRDBNet_arch as arch
# Add the `scripts/` folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))




# Path to the pretrained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'RRDB_ESRGAN_x4.pth')

# Load ESRGAN model
device = torch.device('cpu') #'cuda' if using GPU
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)


def upscale_image(image):
    img = np.array(image) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    return Image.fromarray(output)

# Gradio interface 
def gradio_interface(image):
    try:
        
        if image is None:
            raise ValueError("No image uploaded. Please upload an image to upscale.")

       
        upscaled_image = upscale_image(image)
        original_size = image.size
        upscaled_size = upscaled_image.size

        return image, upscaled_image, f"Original Size: {original_size[0]}x{original_size[1]}", f"Upscaled Size: {upscaled_size[0]}x{upscaled_size[1]}"

    except Exception as e:
        
        return None, None, "Error", str(e)


gr_interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(), gr.Image(), gr.Text(), gr.Text()],
    title="ESRGAN Image Upscaler",
    description="Upload an image to upscale it using ESRGAN."
)

if __name__ == '__main__':
   
    gr_interface.launch()
