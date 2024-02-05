# Automatic License Plate Recognition (ALPR) System

## Overview

This project aims to implement an Automatic License Plate Recognition (ALPR) system using a deep learning model for license plate detection and Optical Character Recognition (OCR) to extract the alphanumeric characters from the detected license plates. The system leverages a dataset from Kaggle for initial training, with an emphasis on expanding the dataset to improve accuracy and robustness.

## System Design

The ALPR system is structured around two main components:

- **License Plate Detection**: Utilizes a deep learning model to identify license plates within vehicle images.
- **OCR for Text Extraction**: Applies OCR technology on the detected plates to read and output the license plate's text.

## Deep Learning Model Architecture

The license plate detection module is a deep learning model built on the EfficientNetB3 architecture. This model was selected for its balance between efficiency and accuracy in image classification tasks. The architecture's depth, characterized by multiple layers designed for feature extraction and pattern recognition, enables it to learn complex visual representations of license plates from diverse backgrounds and conditions.


### Data Challenges

The initial dataset, sourced from Kaggle, is limited in both size and diversity, posing a challenge for training a model to the desired level of accuracy. To address this, we recommend enhancing the dataset with additional images, such as those from the UFPR ALPR dataset, and improving the quality of annotations for more effective training.

![Car Original](Cars94.png)
![Cropped Plate](cropped_Cars94.png)

### Improving the Model

- **Data Augmentation**: Implementing data augmentation techniques to artificially expand the training dataset, helping the model generalize better to unseen data.
- **Exploring Advanced Architectures**: Investigating other deep learning architectures, such as YOLO (You Only Look Once) or SSD (Single Shot MultiBox Detector), for potentially better performance in license plate detection.
- **Custom OCR Model**: Developing a tailored OCR model could significantly enhance the system's ability to accurately read license plate text, especially for plates with non-standard fonts or layouts.

## Usage

1. Train the model using the training script with augmented and enhanced datasets.
2. Detect license plates in new vehicle images using the detection script.
3. Extract text from detected license plates with the OCR script.