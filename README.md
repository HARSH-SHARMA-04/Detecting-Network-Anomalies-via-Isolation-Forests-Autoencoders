This repository contains a machine learning project that includes:

1. **Iris Flower Classification** using a **Random Forest Classifier**.
2. **Anomaly Detection** using two techniques:
   - **Autoencoder** (Deep Learning-based anomaly detection).
   - **Isolation Forest** (Tree-based anomaly detection).

The project uses the **Iris dataset** for classification and the **KDD Cup dataset** for anomaly detection.

## Project Structure

├── app.py # Streamlit app for Iris flower prediction
├── iris_model.pkl # Trained Random Forest model for Iris classification
├── main.py # Main code for training and saving the model
├── requirements.txt # List of required Python libraries
├── README.md # This README file
└── kddcup_data.csv # Sample KDD Cup dataset

markdown
Copy
Edit

## Overview

### 1. Iris Flower Classifier
The project uses the **Iris dataset** to predict the species of Iris flowers. It uses the **Random Forest Classifier** to classify the flowers into three categories:
- Setosa
- Versicolor
- Virginica

### 2. Anomaly Detection
The **KDD Cup dataset** is used for anomaly detection. The project applies two techniques:
- **Autoencoder** (Neural Network-based anomaly detection)
- **Isolation Forest** (Tree-based anomaly detection)

The objective of both techniques is to classify data as `Normal` or `Attack`.

## Prerequisites

Before running the project, you need to install the required libraries. You can do this by installing from the `requirements.txt` file.

### 1. Install Dependencies

To install the required libraries, run the following command in your terminal or command prompt:

```bash
pip install -r requirements.txt
The requirements.txt file includes:

streamlit

scikit-learn

joblib

tensorflow

pandas

numpy

matplotlib

seaborn

How to Use
Step 1: Train and Save the Model
First, run main.py to train the Random Forest classifier on the Iris dataset and save the model:

bash
Copy
Edit
python main.py
This will train the model and save it as iris_model.pkl.

Step 2: Run Streamlit App
The project includes a Streamlit web app for making predictions on Iris flowers. To run the app, use the following command:

bash
Copy
Edit
streamlit run app.py
The app will ask you to input four features (Sepal Length, Sepal Width, Petal Length, and Petal Width), and then predict the Iris species based on these inputs.

Step 3: Anomaly Detection with Autoencoder and Isolation Forest
Autoencoder:
The Autoencoder model is used to detect anomalies in the KDD Cup dataset. It uses reconstruction error to flag anomalous data points.

Isolation Forest:
The Isolation Forest algorithm isolates anomalies based on decision trees. This technique is also applied to the KDD Cup dataset for anomaly detection.

To evaluate the models, the project uses confusion matrices and classification reports.

How to View the Results
After running the app and training the model, the following results will be shown:

Autoencoder Results: Confusion Matrix and Classification Report.

Autoencoder Reconstruction Error: A plot showing the distribution of reconstruction errors, with a threshold for anomaly detection.

Isolation Forest Results: Confusion Matrix and Classification Report.

Isolation Forest Reconstruction Error: A plot showing the anomaly score for the Isolation Forest method.

Example Output
After running the Autoencoder model, you will get:

lua
Copy
Edit
Autoencoder Results:
Confusion Matrix:
[[  0  85]
 [  0 915]]
Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.00      0.00        85
      Attack       0.91      1.00      0.95       915
After running the Isolation Forest model, you will get:

lua

Isolation Forest Results:
Confusion Matrix:
[[ 0  2]
 [ 0 92]]
Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.00      0.00         2
      Attack       0.97      1.00      0.98        92
License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
Harsh Sharma