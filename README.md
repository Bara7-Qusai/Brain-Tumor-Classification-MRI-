
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
```

## Model Architecture

The model uses the ResNet50 architecture, which is pre-trained on ImageNet. The top layer is replaced with a global average pooling layer, followed by batch normalization, dropout, and a dense layer with softmax activation for classification into four categories.

## Training

The model is trained with the following parameters:
- Loss function: Categorical Crossentropy
- Optimizer: Adam with a learning rate of 0.003
- Metrics: Accuracy
- Epochs: 30
- Batch size: 32

Data augmentation is applied to the training images to improve the model's robustness.

## Evaluation

The model is evaluated on a test set, and performance metrics such as classification report and confusion matrix are generated. The model achieved an accuracy of 90%.

## Usage

### Predicting Tumor Type

You can use the widget interface to upload an MRI image and get a prediction. The widget interface allows you to upload an image file and get a prediction of the tumor type.

## Results

The model achieved an accuracy of 90% on the test set. Detailed performance metrics can be found in the generated classification report.



