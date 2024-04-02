# Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution

This repository contains the official implementation of the methods presented in our paper "Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution," currently under review.

## Overview

Our work introduces a novel deep learning framework tailored for endoscopic image super-resolution, with a focus on saliency-aware mechanisms to preserve and highlight critical diagnostic features in the upsampled images.

## Structure

The repository includes the following key components:

- `model.py`: Defines the architecture of the deep learning model.
- `train.py`: Contains the training loop, including data loading, model updates, and logging.
- `test.py`: Code to evaluate the trained model on a test dataset.
- `utils.py`: Utility functions used across the project, such as data preprocessing and metrics computation.

Additionally, several JPEG files are included to illustrate the architecture and performance of our model:

- `Architectures.JPG`
- `Block Diagram.JPG`
- `Downsampled.jpeg`
- `scale_2.jpeg`, `scale_4.jpeg`, `scale_8.jpeg`: Images showcasing the super-resolution results at different scales.

## Installation

To run the code, please install the required libraries as listed in `requirements.txt` (if available). If a `requirements.txt` is not included, please ensure you have a Python environment with the necessary libraries such as TensorFlow or PyTorch, OpenCV, NumPy, etc.

```bash
pip install -r requirements.txt
