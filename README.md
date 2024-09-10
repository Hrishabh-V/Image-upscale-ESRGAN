# Image Upscaling using ESRGAN

This project is a web application for upscaling images using the ESRGAN (Enhanced Super-Resolution Generative Adversarial Networks) model. ESRGAN is a powerful model for image super-resolution, enabling high-quality upscaling of images.

<p align="left">
   <a href="https://huggingface.co/spaces/CHEONMA010/ESRGAN">
      <img alt="LIVE" title=" View the page live on hugging face" src="https://img.shields.io/badge/Hugging_Face-000000?style=for-the-badge&logo=huggingface&logoColor=white"/></a>  
</p>
      
![ESRGAN Example](Diff.png)


## Features
- High-quality image upscaling using ESRGAN.
- Supports both GPU (for faster processing) and CPU (if GPU is not available).

## Dependencies

To run this project, you will need to install the following dependencies:

- CUDA (for GPU acceleration)
- PyTorch
- glob2
- OpenCV

### Installation

1. **Install CUDA**: Follow the instructions on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads) to install CUDA.

2. **Install PyTorch**: You can install PyTorch with CUDA support by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

3. **Install Python packages**:
   ```sh
   pip install glob2 opencv-python

## Note
Before running the application, ensure the ESRGAN model file (`RRDB_ESRGAN_x4.pth`) is downloaded from the provided Google Drive link and placed in the `models` folder.


Download the ESRGAN model file from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY).


This is an intensive program that benefits significantly from GPU acceleration. If a GPU is not available, the application will use the CPU for processing, which will be slower.


## Acknowledgements
This project incorporates code and models from the [ESRGAN project](https://github.com/xinntao/ESRGAN) by Xintao Wang, licensed under the Apache License 2.0.


## License
This project is licensed under the Apache License 2.0.
