# Custom CNN for Fashion MNIST Classification

This repository contains an implementation of a custom Convolutional Neural Network (CNN) for classifying images in the [Fashion MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist). The project includes detailed comparisons of various design choices, such as activation functions, downsampling techniques, and optimizers. The model achieves over 90% accuracy on the test set and adheres to a parameter limit of 1 million trainable parameters.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Comparisons](#comparisons)
  - [Activation Functions](#activation-functions)
  - [Downsampling Methods](#downsampling-methods)
  - [Optimizers](#optimizers)
- [Results](#results)
  - [Performance Metrics](#performance-metrics)
  - [Visualizations](#visualizations)
- [How to Use](#how-to-use)
  - [Installation](#installation)
  - [Running the Script](#running-the-script)
- [Directory Structure](#directory-structure)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project explores the design and implementation of a CNN architecture for the Fashion MNIST dataset. The following features are included:
- A custom CNN with 4 convolutional layers and 2 fully connected layers.
- Use of Batch Normalization and Dropout layers to improve training stability and mitigate overfitting.
- Comparison of:
  - ReLU vs. LeakyReLU activation functions.
  - MaxPooling vs. Strided Convolution for downsampling.
  - Adam optimizer vs. SGD with momentum.
- Model performance visualizations, including training/validation accuracy and loss over epochs.
- Filters and activation map visualizations for interpretability.

---

## Dataset

The [Fashion MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist) consists of 70,000 grayscale images of size 28x28 pixels, divided into:
- 60,000 training images
- 10,000 testing images

Each image belongs to one of the following 10 classes:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

---

## Model Architecture

The CNN architecture includes:
- **Convolutional Layers**: 4 convolutional layers with 32 and 64 filters, kernel size of (3, 3), and padding set to 'same'.
- **Batch Normalization**: Applied after each convolutional operation.
- **Dropout Layers**: Dropout of 25% in convolutional blocks and 50% in dense layers to mitigate overfitting.
- **Fully Connected Layers**: Two dense layers, with the first having 512 units and ReLU/LeakyReLU activation.
- **Output Layer**: A dense layer with 10 units and softmax activation for classification.

Parameter count: **< 1,000,000**

---

## Comparisons

### Activation Functions
- **ReLU**: Faster convergence in early epochs but suffers from the "dying ReLU" problem.
- **LeakyReLU**: Prevents neurons from dying by allowing a small gradient for negative inputs.

### Downsampling Methods
- **MaxPooling**: Reduces spatial dimensions while retaining important features.
- **Strided Convolution**: Learns the downsampling operation but increases parameter count.

### Optimizers
- **Adam**: Adaptive learning rate optimizer that converges faster with minimal tuning.
- **SGD with Momentum**: Slower convergence but can achieve better generalization when tuned properly.

---

## Results

### Performance Metrics
The models were evaluated based on accuracy and loss on both the training and validation sets. All models achieved over 90% test accuracy.

### Visualizations
1. **Training Curves**: Accuracy and loss vs. epochs for each model.
2. **Confusion Matrix**: Displays classification performance for each class.
3. **Filter Visualization**: Shows the filters learned by the first convolutional layer.
4. **Activation Maps**: Visualizes feature activations for a sample input image.

![Training Curves](model_comparison.png)
*Example of training curves comparing different configurations.*

![Confusion Matrix](CNN_ReLU_MaxPool_confusion_matrix.png)
*Confusion matrix for the best-performing model.*

---

## How to Use

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/fashion-mnist-cnn.git
   cd fashion-mnist-cnn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Script
1. Train and evaluate all models:
   ```bash
   python fashion_mnist_cnn.py
   ```

2. Visualize results:
   - Training curves and performance metrics are saved as PNG files.
   - Confusion matrices and activation maps are saved for the best model.

---

## Directory Structure

```
fashion-mnist-cnn/
│
├── fashion_mnist_cnn.py          # Main implementation script
├── visualization_utils.py        # Helper functions for visualizations
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation

```

---

## Acknowledgments

- The [Fashion MNIST dataset](https://www.tensorflow.org/datasets/catalog/fashion_mnist) is provided by Zalando Research.
- TensorFlow and Keras were used for model implementation.
- Special thanks to the open-source community for providing useful resources and tutorials.

---
