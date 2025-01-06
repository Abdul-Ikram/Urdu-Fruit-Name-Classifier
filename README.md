# Handwritten Urdu Fruit Name Classifier

This project focuses on developing a deep learning model to classify handwritten fruit names written in Urdu. Users can upload an image of a handwritten Urdu fruit name, and the model predicts the name of the fruit. The ultimate aim is to provide a robust and interactive application that facilitates learning and exploration for children.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Dataset](#dataset)
3.  [Model Architecture](#model-architecture)
4.  [How to Run](#how-to-run)
5.  [Results](#results)
6.  [Future Work](#future-work)
7.  [Acknowledgments](#acknowledgments)

----------

## Introduction

This project demonstrates the power of convolutional neural networks (CNNs) in image classification tasks, particularly handwritten text in regional languages like Urdu. The model aims to classify 35 unique fruit names, making it a valuable educational tool for young learners.

----------

## Dataset

The dataset used for training and validation consists of handwritten images of 35 different fruit names in Urdu. You can find the dataset on Kaggle:  
**[Handwritten Urdu Fruit Names Dataset](https://www.kaggle.com/datasets/abdulikram/handwritten-text-fruit-classification)**.

-   **Classes:** 35 unique fruit names
-   **Training-Validation Split:** 80%-20%
-   **Images:** High-quality and diverse examples for robust training.

----------

## Model Architecture

The classification model is built using a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. Key features of the architecture include:

-   Multiple Conv2D and MaxPooling2D layers
-   Dropout layers for regularization
-   Activation functions: ReLU and Softmax
-   Optimizer: Adam
-   Loss function: Categorical Crossentropy

----------

## How to Run

1.  Clone the repository:
    
    ```bash
    git clone https://github.com/Abdul-Ikram/Urdu-Fruit-Name-Classifier.git
    cd Urdu-Fruit-Name-Classifier
    ```
    
2.  Open the Jupyter Notebook:
    
    ```bash
    jupyter notebook
    ```
    
3.  Follow the steps in the notebook to:
    
    -   Preprocess the dataset.
    -   Train the model.
    -   Evaluate the model's performance.
4.  To classify an image:
    
    -   Modify the input section in the notebook to provide the path to your handwritten Urdu fruit name image.
    -   Run the classification cell to see the predicted fruit name.

----------

## Results

The model was trained for 150 epochs, achieving significant accuracy on the validation set. Metrics and visualizations of training/validation performance are included in the notebook.

----------

## Future Work

-   Extend classification to include vegetables, flowers, and animals.
-   Deploy the model as a web application for easy access and interactivity.
-   Improve the dataset by adding more diverse examples to enhance generalization.

----------

## Acknowledgments

-   **Dataset:** [Kaggle](https://www.kaggle.com/datasets/abdulikram/handwritten-text-fruit-classification)
-   Frameworks: TensorFlow, Keras, OpenCV
-   Tools: Jupyter Notebook, Matplotlib
