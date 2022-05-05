# Emotion Recognition Web Application with Keras, OpenCV, and Flask

**Table of Contents**:

<!--ts-->
- [Introduction](#introduction)
- [Overview](#overview)
  - [Methods Used](#methods-used)
  - [Technologies](#technologies)
  - [Libraries](#libraries)
- [Data](#data)
- [References](#references)
- [Feedback](#feedback)
<!--te-->

<br>

## Introduction

In this project, we will create a web app that detects human faces in a frame (image or video) and classifies them based on emotion. The project is based on three main pillars/axes:

- Building and training a Convolutional Neural Network with Keras
- Implementing face detection using OpenCV
- Hosting the app in a browser using Flask

<br>

## Overview:

As mentioned earlier, our project can be segmented into three different sections:

1)	**Building and Training a Convolutional Neural Network** <br>
This section is performed entirely in the Jupyter notebook run on Google Colab. It contains the usual steps in training a DNN model: loading the data, performing data augmentation, creating, compiling, and training the model, and using the trained model to make predictions. The full model architecture along with a schematic are included in the Images folder.
  
2)	**Face Detection** <br>
This part of the project will be implemented using Haar cascades and OpenCV. A Haar classifier, or a Haar cascade classifier, is an object detection program that identifies objects in an image or video. The OpenCV library maintains a repository of pre-trained Haar cascades. We only need the ` haarcascade_frontalface_default.xml` file.
  
3)	**Hosting the App** <br>
Lastly, we will use Python’s Flask web framework to host our application in a browser. For this purpose, the app.py file loads the DL model and the Haar cascade classifier, detects a face and uses the model to predict emotion.

<br>

### Methods Used
- Image Classification with Convolutional Neural Networks
- Computer Vision (Face Detection)
- Model Deployment
- Data Visualisation
- Data Augmentation

<br>

### Technologies
- Python
- Jupyter Notebook
- Google Colab

<br>

### Libraries
- Keras
- Tensorflow
- Flask
- Cv2
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn

<br>

## Data

This project requires two datasets:

- **[`FER2013`](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)**: A well-studied dataset that has been the subject of many Machine Learning competitions and research papers. The dataset contains 35887 images normalized to 48x48 pixels in grayscale and organized into different folders based on the emotion they depict. There are seven different emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

- **[`CK+`](https://paperswithcode.com/dataset/ck)**: The Extended Cohn-Kanade (CK+) dataset contains 593 image sequences from a total of 123 different subjects, ranging from 18 to 50 years of age, with a variety of genders and heritage. Even though the number of images (981) is small compared to FER2013, including those images adds some diversity to the data and leads to a small increase in accuracy.

Both datasets are extracted from [Kaggle](https://www.kaggle.com/). Instructions on how to download them are provided in the [Jupyter notebook](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/Emotion_Recognition_Notebook.ipynb).


<br>

## References

The main resources I used are the following two books:

[1] Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems*. 2nd ed., O’Reilly Media, 2019.

[2] Chollet, Francois. *Deep Learning with Python*. 2nd ed., Manning, 2021.

<br>

The following resources also helped me in my analysis:



<br>

## Feedback

If you have any feedback or ideas to improve this project, feel free to contact me via:

<a href="https://twitter.com/korfanakis">
  <img align="left" alt="Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />
</a>

<a href="https://uk.linkedin.com/in/korfanakis">
  <img align="left" alt="LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />
</a>
