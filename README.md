# PrivSeg

Medical image segmentation using deep learning models for skin lesion analysis.

## Overview

This project implements image segmentation models for the HAM10000 dataset, focusing on skin lesion segmentation. It includes implementations of UNet and Mask2Former architectures with training, inference, and model comparison utilities.

## Project Structure

```
src/
├── model.py              # UNet architecture
├── model_mask2former.py  # Mask2Former architecture
├── dataset.py            # HAM10000 dataset loader
├── train.py              # Training script for UNet
├── train_mask2former.py  # Training script for Mask2Former
├── inference.py          # Inference utilities for UNet
├── inference_mask2former.py  # Inference utilities for Mask2Former
└── compare_models.py     # Model comparison and evaluation
```

## Features

- **UNet Model**: Classic encoder-decoder architecture for medical image segmentation
- **Mask2Former**: Advanced segmentation model with improved accuracy
- **Dataset Handling**: Custom dataset loader for HAM10000 skin lesion images
- **Training**: Complete training pipelines with Dice loss optimization
- **Inference**: Easy-to-use inference scripts for predictions
- **Model Comparison**: Tools to evaluate and compare model performance

## Usage

### Training

```bash
python src/train.py              # Train UNet model
python src/train_mask2former.py  # Train Mask2Former model
```

### Inference

```bash
python src/inference.py              # Run UNet inference
python src/inference_mask2former.py  # Run Mask2Former inference
```

### Model Comparison

```bash
python src/compare_models.py  # Compare model performance
```

## Requirements

- PyTorch
- torchvision
- PIL (Pillow)
- tqdm
- numpy

## Dataset

The project uses the HAM10000 dataset containing dermoscopic images and corresponding segmentation masks for skin lesions.
