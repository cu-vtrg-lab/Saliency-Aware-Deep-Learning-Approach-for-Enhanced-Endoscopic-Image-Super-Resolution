# Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution

This repository contains the official implementation of the methods presented in our paper "**Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution**" currently under review.
<p align="center">
  <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/Block%20Diagram.JPG" width="100%">
</p>

## Overview
Our work introduces a novel deep learning framework tailored for endoscopic image super-resolution, with a focus on saliency-aware mechanisms to preserve and highlight critical diagnostic features in the upsampled images.

## Requirements
- **PyTorch 1.3.0, torchvision 0.4.1**: The code is tested with `python=3.7, cuda=9.0`.
- **Matlab**: For training/test data generation and performance evaluation.

## Train
- **Data Preparation**: Download the training sets from (https://endovis.grand-challenge.org) for SCARED dataset unzip this into `./data/train/`.
- **Generate Patches**: Run `./data/train/GenerateTrainingPatches.m` in Matlab to generate training patches.
- **Training**: Execute `python train.py` to start the training process. The checkpoints will be saved to `./log/`.

## Test
- **Data Preparation**: Acquire the complete test sets:
  - **SCARED dataset**: Available at [EndoVis Grand Challenge](https://endovis.grand-challenge.org).
  - **MICCAI 2017 Kidney Boundary Detection SubChallenge dataset**:  Available at [EndoVis Grand Challenge](https://endovis.grand-challenge.org).
  - **MICCAI 2017 Robotic Instrument Segmentation Sub-Challenge dataset**:  Available at [EndoVis Grand Challenge](https://endovis.grand-challenge.org).
  - **MICCAI 2019 Challenge on Stereo Correspondence and Reconstruction of Endoscopic Data dataset**:  Available at [EndoVis Grand Challenge](https://endovis.grand-challenge.org).
  - **Da Vinci dataset**: Accessible at [GitHub - hgfe/DCSSR](https://github.com/hgfe/DCSSR).
  
  After downloading, unzip the datasets into the `./data` directory.

- **Inference**: Execute `python test.py` to perform a demo inference. The `.png` files showcasing the results will be stored in the `./results` folder.

- **Evaluation**: Run `evaluation.m` in Matlab to calculate the PSNR and SSIM metrics, which are standard for assessing image quality.

## Quantitative Results

  <p align="center">
    <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/Capture.JPG" width="100%">
  </p>

## Assessment of the Visual Quality of SR Images Created Through Image Super-Resolution Techniques

- ** On ×2 Scale Factor on the SCARED Dataset:**
  <p align="center">
    <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/scale_2.jpeg" width="100%">
  </p>

- **On ×4 Scale Factor on the Robotic Instrument Segmentation Dataset:**
  <p align="center">
    <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/scale_4.jpeg" width="100%">
  </p>

- **On ×8 Scale Factor on the Stereo Correspondence and Reconstruction Dataset:**
  <p align="center">
    <img src="https://github.com/MansoorHayat777/StereoEndoscopicImageSuper-Resolution-SEISR/raw/main/SEISR/scale_8.jpeg" width="100%">
  </p>


