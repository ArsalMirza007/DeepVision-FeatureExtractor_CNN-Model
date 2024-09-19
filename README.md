# Cat and Dog Image Recognition with Deep Learning

This project implements a Convolutional Neural Network (CNN) for image recognition using TensorFlow and Keras. The CNN model is trained to classify images as either cats or dogs. The project includes a Streamlit web application for user-friendly interaction and real-time image classification.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Acknowledgements](#acknowledgements)

## Introduction

This project demonstrates how to build a CNN model for image classification using deep learning techniques. It provides an end-to-end solution from training the model on a dataset of cat and dog images to deploying an interactive web application that allows users to upload images and receive predictions.

## Features

- **Deep Learning Model**: A CNN built with TensorFlow and Keras for binary image classification (cats vs. dogs).
- **Data Augmentation**: Enhances the model's robustness by applying transformations such as rescaling, shearing, zooming, and flipping.
- **Model Saving**: The trained model is saved in `.h5` format for easy loading and reuse.
- **Interactive Web Interface**: Built with Streamlit to allow users to upload images and view predictions in real-time.
- **Result Visualization**: Training and validation metrics are visualized using Matplotlib and Seaborn.

## Installation

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Streamlit
- Other dependencies listed in `requirements.txt`

### Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/image-recognition-cnn.git
    cd image-recognition-cnn
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download or Prepare Dataset:**
    - Organize your dataset in the following directory structure:

      ```
      dataset/
      ├── training_set/
      │   ├── cats/
      │   └── dogs/
      └── test_set/
          ├── cats/
          └── dogs/
      ```

## Usage

### Training the Model

To train the CNN model, run the following command:

```bash
python train.py
```
This will train the model using the dataset and save it as saved_model/my_cnn_model.h5.

## Running the Streamlit Application
To start the Streamlit web application, use the following command:

```bash
python -m streamlit run app.py
```
This command will launch the Streamlit app, which you can access in your web browser at http://localhost:8501.

## Uploading and Classifying an Image
- Open the web interface and upload an image file (JPG, JPEG, or PNG).
- The application will display the uploaded image and predict whether it is a cat or a dog.
  
## Dataset
- **Training Set:** Contains images of cats and dogs used to train the model.
- **Test Set:** Contains images of cats and dogs used to evaluate the model's performance.

## Model Training
### The CNN model architecture includes:
- **Input Layer:** Image input of size 64x64x3 (width, height, RGB channels).
- **Convolutional Layers:** Two convolutional layers with ReLU activation and max pooling.
- **Flattening Layer:** Converts 2D matrix data into a 1D vector.
- **Fully Connected Layer:** A dense layer with ReLU activation.
- **Output Layer:** A single neuron with sigmoid activation for binary classification.

## Model Evaluation
The model's performance is evaluated using accuracy and loss metrics on the test set. Training and validation metrics are visualized using Matplotlib and Seaborn.

## Model Deployment
The trained model is deployed using Streamlit, providing an interactive web interface for users to upload images and get predictions.

## Acknowledgements
- Thank you to the TensorFlow and Keras teams for their deep learning frameworks.
- Thanks to Streamlit for providing a straightforward way to build interactive web applications.

# Screenshorts
- **Traing and Loss Visualization:**
  ![Screenshot (110)](https://github.com/user-attachments/assets/82faa7b0-2b9a-4aa0-8e2f-82aa486600ac)
  
- **App Running:**
  ![Screenshot (111)](https://github.com/user-attachments/assets/eab5fa5d-404d-44ca-a257-8d7025286407)

- **Select Image:**
  ![Screenshot (112)](https://github.com/user-attachments/assets/315a1dbf-6db1-45dd-a0f1-38296309d3ba)

- **Dog Prediction:**
  ![Screenshot (113)](https://github.com/user-attachments/assets/18bcabf7-3e98-4718-9cde-1b67e971bb7b)
  ![Screenshot (114)](https://github.com/user-attachments/assets/f94a77fe-fead-49bf-ad43-1968933b0890)

- **Cat Prediction:**
  ![Screenshot (115)](https://github.com/user-attachments/assets/af2a3f00-f86c-4702-8958-05ed17738d68)
  ![Screenshot (116)](https://github.com/user-attachments/assets/0eaf7475-c9bd-45c2-9c5e-e9cf0270b103)

