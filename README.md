# Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution

This repository contains the official implementation of the methods presented in our paper "Saliency-Aware Deep Learning Approach for Enhanced Endoscopic Image Super-Resolution," currently under review.

## Abstract

The adoption of Stereo Imaging technology within endoscopic procedures represents a transformative advancement in medical imaging, providing surgeons with depth perception and detailed views of internal anatomy for enhanced diagnostic accuracy and surgical precision. However, the practical application of stereo imaging in endoscopy faces challenges, including the generation of low-resolution and blurred images, which can hinder the effectiveness of medical diagnoses and interventions. Our research introduces an endoscopic image super-resolution model in response to these specific. This model features an innovative feature extraction module and an advanced cross-view feature interaction module tailored for the intricacies of endoscopic imagery. Initially trained on the SCARED dataset, our model was rigorously tested across four additional publicly available endoscopic image datasets at scales 2, 4, and 8, demonstrating unparalleled performance improvements in endoscopic super-resolution. Our results are compelling, showing that our model not only substantially enhances the quality of endoscopic images but also consistently surpasses other existing methods in all tested datasets, in quantitative measures such as PSNR and SSIM, and qualitative evaluations. The successful application of our super-resolution model in endoscopic imaging has the potential to revolutionize medical diagnostics and surgery, significantly increasing the precision and effectiveness of endoscopic procedures.
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

