# MNIST Digit Classifier (PyTorch)

A Convolutional Neural Network (CNN) built from scratch in PyTorch to classify handwritten digits from the MNIST dataset. The project implements the complete deep learning pipeline, from loading raw IDX files to training and evaluating the model.

## Features

- Custom MNIST IDX file parser
- Custom PyTorch `Dataset`
- CNN implemented from scratch
- Manual training and evaluation loops
- Achieves **98.11%** test accuracy

## Architecture

```
Input (1×28×28)
    ↓
Conv2D (1 → 32)
    ↓
ReLU
    ↓
MaxPool
    ↓
Conv2D (32 → 64)
    ↓
ReLU
    ↓
MaxPool
    ↓
Flatten
    ↓
Linear (1600 → 512)
    ↓
ReLU
    ↓
Linear (512 → 128)
    ↓
ReLU
    ↓
Linear (128 → 10)
```

## Tech Stack

- Python
- PyTorch
- NumPy

## Run

```bash
pip install torch numpy
python train.py
```
