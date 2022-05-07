# Emotion Recognition Web Application with Keras, OpenCV, and Flask

<p align="center">
  <img src="images/Intro.gif" style="width: 400px;"/>
</p>

**Table of Contents**:

<!--ts-->
- [Introduction](#introduction)
- [Overview](#overview)
- [Data](#data)
- [Running the App](#running-the-app)
- [References](#references)
- [Feedback](#feedback)
<!--te-->

<br>

## Introduction

In this project, we will create a web app that detects human faces in a frame (image or video) and classifies them based on emotion. The project consists of three main sections:

- Building and training a Convolutional Neural Network (CNN) with Keras
- Implementing face detection using OpenCV
- Hosting the app in a browser using Flask

<br>

**→ Skills**: *Image Classification with Convolutional Neural Networks, Computer Vision (Face Detection), Model Deployment, Data Visualisation, Data Augmentation* <br>
**→ Technologies**: *Python, Jupyter Notebook, Google Colab* <br>
**→ Libraries**: *Keras, Tensorflow, Flask, OpenCV(cv2), Scikit-learn, Numpy, Matplotlib, Seaborn* <br>

<br>

## Overview:

###	**1. Building and Training a CNN** <br>

This section is performed entirely in the Jupyter notebook run on Google Colab. It contains the usual steps in training a DNN model: loading the data, performing data augmentation, creating, compiling, and training the model, and using the trained model to make predictions. A schematic illustration of the model’s architecture is shown below. The model achieves aproximately 69% accuracy across all labels/emotions. For a more detailed breakdown of the model's performance, please refer to the notebook. 

<p align="center">
  <img src="images/CNN_Architecture.svg" style="width: 700px;"/>
</p>

<br>

###	**2. Face Detection** <br>

This part of the project will be implemented using [Haar Cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) and [OpenCV](https://opencv.org/). A Haar classifier, or a Haar cascade classifier, is an object detection program that identifies objects in an image or video. The OpenCV library maintains a repository of pre-trained Haar cascades. For this project, we only need the `haarcascade_frontalface_default.xml` file, which detects the front of human faces.

<br>

### **3. Hosting the App** <br>

Lastly, we will use Python’s [Flask](https://flask.palletsprojects.com/en/2.1.x/) web framework to host our application in a browser. For this purpose, the [`app.py`](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/app.py) file loads the CNN model and the Haar cascade classifier, detects a face and uses the model to predict its emotion.

<br>

## Data

For training and testing the mode, we will use the **[`FER2013`](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)** dataset, a well-studied dataset that has been the subject of many Deep Learning competitions and research papers. The dataset consists of 35.887 images of human faces normalized to 48x48 pixels in grayscale and organised into different folders based on the emotion they depict. There are seven different emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`. Unfortunately, there is a significant imbalance in the dataset, with the `happy` class being the most prevelant and the `disgust` class being noticeably underrepresented.

The dataset is extracted from Kaggle through [this link](jonathanoheix/face-expression-recognition-dataset). Instructions on how to download it and open it with Colab are provided in the [notebook](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/Emotion_Recognition_Notebook.ipynb).

<br>

<p align="center">
  <img src="images/Emotions.svg " style="width: 1000px;"/>
</p>

<br>

## Running the App

The most straightforward way to launch the Flask app is to run it locally. The most straightforward way to launch the Flask app is to run it locally. First, open a command-line prompt and navigate to the project’s directory and run the following commands:

```
set FLASK_APP=app
flask run
```

The `FLASK_APP` environment variable is the name of the module to import at flask run. The `flask` command is installed by Flask; it must be told where to find your application in order to use it. 



<br>

## References

The main resources I used are the following two books:

[1] Géron, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems*. 2nd ed., O’Reilly Media, 2019.

[2] Chollet, Francois. *Deep Learning with Python*. 2nd ed., Manning, 2021.

<br>

## Feedback

If you have any feedback or ideas to improve this project, feel free to contact me via:

<a href="https://twitter.com/korfanakis">
  <img align="left" alt="Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/twitter.svg" />
</a>

<a href="https://uk.linkedin.com/in/korfanakis">
  <img align="left" alt="LinkedIn" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />
</a>
