# Deep CNN Image Classification for Website Page Capture Detection

## Introduction

This project focuses on classifying website page captures into Front Page or Checkout Page categories using a deep convolutional neural network (CNN). I built this project after my Algorithms and Programming professor proposed me to get involved into the AI and ML research conducted by Madalina Dicu, a Ph.D. candidate at my University, in order to geat familiarised with research, different thematics, and machine learning. 

## Motivation

The research's focus was building an AI model that could classify all elements(logic, buttons, usage, etc.) from a website page, regardless of indications or prior exposure to it. Because I was interested in CNNs and Machine Learning, Ms. Dicu proposed to build a simple Deep CNN classifier(related to the research), in order to classify Front Page or Checkout Page categories, to explore machine learning techniques.

## Instalation

Clone the repository:

```bash
git clone https://github.com/MihaiBogdanDeaconu/Deep-CNN-Website-Page-Classifier
```

Install dependencies:

```bash
pip install tensorflow tensorflow-gpu opencv-python matplotlib
```

## Project Structure

### 1.Environment Setup

Before running the code, ensure GPU memory growth for TensorFlow to avoid memory allocation issues.

### 2.Remove Bad Images

Remove images with unsupported file extensions to clean the dataset.

### 3.Load Data

Load the dataset using TensorFlow's `image_dataset_from_directory` function and split it into training, validation, and testing sets.

### 4.Scale Data

Scale the pixel values of images to the range [0, 1].

### 5.Split Data

Split the dataset into training, validation, and testing sets.

### 6.Build Deep Learning Model

Construct a CNN model using TensorFlow's Keras API. This involves:

- **Convolutional Layers**: These layers apply filters to input images to extract features such as edges, textures, or patterns.

- **Max-Pooling Layers**: These layers downsample the feature maps, reducing computational complexity and extracting the most important features.

- **Flatten Layer**: This layer converts the multi-dimensional feature maps into a one-dimensional vector for input to the dense layers.

- **Dense Layers**: These fully connected layers process the flattened feature vector to make predictions. Activation functions like ReLU are used to introduce non-linearity, while the output layer typically uses a sigmoid activation for binary classification.

### 7.Train Model

Train the model on the training set, validate it on the validation set, and monitor performance using TensorBoard.

### 8.Plot Performance

Visualize the training and validation loss, as well as accuracy over epochs.

### 9.Evaluate Model

Evaluate the model's performance on the testing set using metrics such as precision, recall, and accuracy.

### 10.Test Model

Test the trained model on a sample image to predict whether it belongs to the Front Page or Checkout Page category.

### 11.Save Model

Save the trained model for future use and load it back to ensure successful model persistence.

## Training Data

The training data can be found in the "Data" folder.

## Model Highlights

File Structure:

![File Structure](Model%20Highlights/Data%20GCollab%20Structure.JPG)

Training:

![File Structure](Model%20Highlights/Training.JPG)

Accuracy:

![File Structure](Model%20Highlights/Accuracy.JPG)

Loss:

![File Structure](Model%20Highlights/Loss.JPG)


## Dependencies

- TensorFlow
- OpenCV
- Matplotlib


## Author

The research was conducted by Madalina Dicu who also supervised me for this project.

## Contributions

Feel free to contribute to and extend this project for further research and applications in machine learning and image classification.



