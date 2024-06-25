# Copyright [2024] [Hrishabh V]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from flask import Flask, request, render_template, send_file
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu') # use CUDA when using GPU

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
    
    return output

# Custom Jinja filters
def b64encode(data):
    return base64.b64encode(data).decode('utf-8')

app.jinja_env.filters['b64encode'] = b64encode

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            image = Image.open(file)
            upscaled_image = upscale_image(image)
            upscaled_image_pil = Image.fromarray(upscaled_image)

            original_img_io = io.BytesIO()
            upscaled_img_io = io.BytesIO()
            image.save(original_img_io, 'PNG')
            upscaled_image_pil.save(upscaled_img_io, 'PNG')
            original_img_io.seek(0)
            upscaled_img_io.seek(0)
            
            return render_template('index.html', 
                                   original_image=original_img_io.getvalue(), 
                                   upscaled_image=upscaled_img_io.getvalue(),
                                   original_size=image.size,
                                   upscaled_size=upscaled_image_pil.size)
    return render_template('index.html')

@app.route('/download/<filename>', methods=['GET'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
