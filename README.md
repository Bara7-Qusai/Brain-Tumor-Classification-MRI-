# Brain Tumor Classification using ResNet50

This repository contains code for classifying brain tumors using a Convolutional Neural Network (CNN) based on the ResNet50 architecture. The model classifies brain MRI images into four categories: glioma tumor, no tumor, meningioma tumor, and pituitary tumor.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Introduction

This project uses a deep learning approach to classify brain tumors from MRI images. The dataset is split into training and testing sets, and data augmentation is applied to enhance the training data. The ResNet50 model, pre-trained on ImageNet, is fine-tuned for this classification task.

## Dataset

The dataset used in this project contains MRI images classified into four categories:
- Glioma Tumor
- No Tumor
- Meningioma Tumor
- Pituitary Tumor

The images are resized to 150x150 pixels before being fed into the model.

## Dependencies

The following libraries are required to run the code:
- matplotlib
- numpy
- pandas
- seaborn
- opencv-python
- tensorflow
- tqdm
- scikit-learn
- ipywidgets
- Pillow

You can install the necessary packages using the following command:
```bash
pip install matplotlib numpy pandas seaborn opencv-python tensorflow tqdm scikit-learn ipywidgets Pillow
