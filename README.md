# Emotion Recognition Web Application with Keras, OpenCV, and Flask

<p align="center">
  <img src="images/Intro.gif" style="width: 300px;"/>
</p>

**Table of Contents**:

<!--ts-->
- [Overview](#overview)
- [Emotion Recognition](#emotion-recognition)
- [Business Objective](#business-objective)
- [Project Summary](#project-summary)
  - [1. Building and Training a CNN](#1-building-and-training-a-cnn)
  - [2. Face Detection](#2-face-detection)
  - [3. Hosting the App](#3-hosting-the-app)
- [Data](#data)
- [Running the App](#running-the-app)
- [References](#references)
- [Feedback](#feedback)
<!--te-->

<br>

## Overview

In this project, we will create a web app that detects human faces in a frame (image or video) and classifies them based on emotion. The project consists of three main sections:

- Building and training a Convolutional Neural Network (CNN) with Keras
- Implementing face detection using OpenCV
- Hosting the app in a browser using Flask

<br>

**→ Skills**: *Image Classification with Convolutional Neural Networks, Computer Vision (Face Detection), Model Deployment, Data Visualisation, Data Augmentation* <br>
**→ Technologies**: *Python, Jupyter Notebook, Google Colab* <br>
**→ Libraries**: *Keras, Tensorflow, Flask, OpenCV (cv2), Scikit-learn, Numpy, Matplotlib, Seaborn* <br>

<br>

## Emotion Recognition

Facial Emotion Recognition (FER) is the process of detecting displayed human emotions using artificial intelligence based technologies in order to evaluate non-verbal responses to products, services or goods.

This is important, as computer systems can adapt their responses and behavioural patterns according to the emotions of humans, thus making the interaction more natural. One application can be found in an automatic tutoring system, where the system adjusts the level of the tutorial depending on the user's affective state, such as excitement or boredom.

Additionally, businesses can use FER to gain additional feedback on products and services. Using facial emotion recognition can aid in understanding which emotions a user is experiencing in real-time. This is a great addition to verbal feedback as it provides a more complex review of the user experience.
Consequently, FER has been an active field of computer vision and has use cases across a variety of industries, such as healthcare, marketing, manufacturing, etc.

<br>

## Business Objective

We are hired as data scientists by an advertising company, specialising in electronic boards at football matches. The company wants to develop software that detects fan’s faces, estimates their emotion, and adjusts ads based on the collective emotion.

Our task is to develop a Deep Learning model that implements emotion recognition and integrates it with a face detection algorithm. The final product shall be delivered as a web application that accepts live video as input. The company will then integrate our product with their systems for automatic ad renewal according to the change of emotions during football games.


<br>

## Project Summary

###	1. Building and Training a CNN

This section is performed entirely in the [Jupyter notebook](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/Emotion_Recognition_Notebook.ipynb) run on Google Colab. It contains the usual steps in training a DNN model: loading the data, performing data augmentation, creating, compiling, and training the model, and using the trained model to make predictions. The dataset used to train the CNN is the FER2013 dataset (more details are provided in the [Data](#data) section). A schematic illustration of the model’s architecture is shown below, while a more detailed summary (as produced by the `summary()` method) is included in the [images folder](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/images/CNN_Architecture_Summary.png).

<p align="center">
  <img src="images/CNN_Architecture.svg" style="width: 700px;"/>
</p>

The model achieves **aproximately 69.5% (validation) accuracy** across all labels/emotions, beating the **baseline human-level accuracy of ~65%**. For a more detailed breakdown of the model's performance, please refer to the [Assessing Performance](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/Emotion_Recognition_Notebook.ipynb) section of the Jupyter notebook. 

<br>

###	2. Face Detection

This part of the project will be implemented using [Haar Cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) and [OpenCV](https://opencv.org/). A Haar classifier, or a Haar cascade classifier, is an object detection program that identifies objects in an image or video. The OpenCV library maintains a repository of pre-trained Haar cascades. For this project, we only need the `haarcascade_frontalface_default.xml` file, which detects the front of human faces.

<br>

### 3. Hosting the App

Lastly, we will use Python’s [Flask](https://flask.palletsprojects.com/en/2.1.x/) web framework to host our application in a browser. For this purpose, the [`app.py`](https://github.com/KOrfanakis/Emotion_Recognition_Deep_Learning_App/blob/main/app.py) file loads the CNN model and the Haar cascade classifier, detects a face and uses the model to predict its emotion. The HTML document used to create the web app is included in the templates folder. Instructions on how to run the web app are provided in the [Running the App](#running-the-app) section.

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

The most straightforward way to launch the Flask app is to run it locally. First, open a command-line prompt, navigate to the project’s directory and run the following commands:

```
set FLASK_APP=app
```
Followed by:
```
flask run
```

Finally, open up a web browser and enter the following URL in the address field:
```
http://localhost:5000/
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
