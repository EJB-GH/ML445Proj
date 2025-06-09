# Alzheimer MRI Classification Project

**Course:** PSU CS445 - Machine Learning (Final Assignment)

---

## Overview

This project explores classification of MRI images into four categories representing stages of Alzheimer’s disease using two models:

- A basic Multi-Layer Perceptron (MLP)  
- A Convolutional Neural Network (CNN)

The goal is to compare these models’ performances on the same dataset and gain experience with CNNs, which are expected to outperform the MLP.

**Dataset:** [Alzheimer MRI 4 Classes Dataset on Kaggle](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset/data)

---

## Dependencies

The required Python packages are listed in `requirements.txt`. To set up your environment, run:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- torch
- torchvision
- matplotlib
- scikit-learn

---

## Usage

... need to add here ...



---

## Notes

- The MLP is a manually implemented baseline neural network using PyTorch tensors and functions.
- The CNN uses PyTorch’s nn.Module and standard layers for improved spatial feature extraction.
- CNN performance is expected to be better due to its convolutional architecture.
- See the source code in mlp.py and cnn.py for detailed implementation.



