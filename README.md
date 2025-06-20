# END-TO-END-DATA-SCIENCE-PROJECT:

**COMPANY**  : CODTECH IT SOLUTIONS

**NAME**     : POLASANI PURUSHOTHAM REDDY

**INTERN ID**: CT08DN1100

**DOMAIN**   : DATA SCIENCE

**DURATION** : 8 WEEEKS

**MENTOR**   : NEELA SANTOSH




# 🌸 Iris Flower Classification – End-to-End Data Science Project

This repository contains a complete end-to-end data science project that demonstrates how to go from **data collection and preprocessing** to **model training, saving, and deployment** using **Flask**. The project is based on the popular **Iris Flower Classification** problem using a machine learning model built with `scikit-learn`.It covers everything from data collection and preprocessing to training a model and deploying it using a web framework. In this case, we use Flask to build a web application that lets users input values and receive predictions about the species of an iris flower.

This application is ideal for beginners and intermediate learners who are interested in how machine learning can be integrated into real-world applications. It walks through how to prepare and scale data, train a model using scikit-learn, save the model using pickle, and create a user interface using Flask.
The web application allows users to input four features related to a flower (sepal length, sepal width, petal length, and petal width), and it responds with a predicted flower species. The prediction is generated in real time using a trained machine learning model running in the background.
## 📚 Dataset: Iris

The **Iris dataset** is a well-known classification dataset in the machine learning community. It contains 150 instances of iris flowers, each described by four numerical features:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Each flower belongs to one of three species:
- **Setosa**
- **Versicolor**
- **Virginica**

This dataset is clean, well-structured, and perfect for learning and demonstration purposes.


## 🚀 Project Overview

This project walks you through the process of:

1. Loading and preparing the Iris dataset
2. Preprocessing the data with `StandardScaler`
3. Training a machine learning model (`RandomForestClassifier`)
4. Saving the trained model and scaler using `pickle`
5. Creating a web interface using **Flask**
6. Deploying the model as a web application with an HTML form

It provides a simple, educational, and production-ready structure that you can reuse for your own ML deployment projects.

## 🧠 Machine Learning Pipeline

The pipeline follows these steps:

1. **Data Loading**: Use `sklearn.datasets.load_iris()` to import the Iris dataset.
2. **Data Splitting**: Use `train_test_split` to separate the data into training and test sets.
3. **Feature Scaling**: Apply `StandardScaler` to normalize the features.
4. **Model Training**: Train a `RandomForestClassifier`, a robust and efficient classifier.
5. **Serialization**: Save the trained model and scaler using `pickle`.

## 🗂️ Project Structure

├── app.py                   # Flask app with prediction endpoint

├── model.pkl                # Trained model

├── scaler.pkl                # Trained scaler
  
└── index.html                # HTML form for user input

├── train_model.py            # Script to train and save the model

├── requirements.txt          # Python dependencies

└── README.md                 # Description file

# OUTPUTS:

![Image](https://github.com/user-attachments/assets/58330a99-5864-43a9-9d05-cbb666f0b4da)


![Image](https://github.com/user-attachments/assets/25f532d6-ad41-4b37-a738-4ab9faa9c0cf)
