# Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution

This repository contains the official implementation of the methods presented in our paper "Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution," currently under review.
<p align="center">
  <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/Block%20Diagram.JPG" width="100%">
</p>

## Abstract
Our work introduces a novel deep learning framework tailored for endoscopic image super-resolution, with a focus on saliency-aware mechanisms to preserve and highlight critical diagnostic features in the upsampled images.
## Codes and Models

### Requirements
- **PyTorch 1.3.0, torchvision 0.4.1**: The code is tested with `python=3.7, cuda=9.0`.
- **Matlab**: For training/test data generation and performance evaluation.

### Train
- **Data Preparation**: Download the training sets from (https://endovis.grand-challenge.org) for SCARED dataset unzip this into `./data/train/`.
- **Generate Patches**: Run `./data/train/GenerateTrainingPatches.m` in Matlab to generate training patches.
- **Training**: Execute `python train.py` to start the training process. The checkpoints will be saved to `./log/`.

### Test
- **Data Preparation**: Download the full test sets the SCARED dataset, the MICCAI 2017 Kidney Boundary Detection SubChallenge; the Kidney Boundary Detection dataset, the MICCAI 2017 Robotic Instrument Segmentation Sub-Challenge; Robotic Instrument Segmentation, and the MICCAI 2019 challenge on Stereo Correspondence and Reconstruction of Endoscopic Data; Stereo Correspondence and Reconstruction from (https://endovis.grand-challenge.org) and da Vinci dataset from (https://github.com/hgfe/DCSSR) unzip them into `./data`.
- **Inference**: Run `python test.py` to perform a demo inference. The resulting images (`.png` files) will be saved in `./results`.
- **Evaluation**: Execute `evaluation.m` in Matlab to compute the PSNR and SSIM scores.


Additionally, several JPEG files are included to illustrate the architecture and performance of our model:

- `Architectures.JPG`
- `Block Diagram.JPG`
- `Downsampled.jpeg`
- `scale_2.jpeg`, `scale_4.jpeg`, `scale_8.jpeg`: Images showcasing the super-resolution results at different scales.

## Installation

To run the code, please install the required libraries as listed in `requirements.txt` (if available). If a `requirements.txt` is not included, please ensure you have a Python environment with the necessary libraries such as TensorFlow or PyTorch, OpenCV, NumPy, etc.

