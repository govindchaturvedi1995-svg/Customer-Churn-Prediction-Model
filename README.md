# 📊 Customer Churn Prediction System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

<p align="center">

An end-to-end Machine Learning application that predicts customer churn using demographic and service information. The project compares multiple machine learning models, automatically selects the best-performing model, and deploys it through an interactive Streamlit web application.

</p>

---

# 🚀 Live Demo

> **Live Application:**  
> **🔗 Coming Soon...** *(Paste your deployed Streamlit URL here after deployment.)*

Example:

```
https://your-app-name.streamlit.app
```

---

# 🌟 Project Highlights

- 🤖 Trains **two Machine Learning models** (Random Forest & Artificial Neural Network).
- 📈 Automatically compares both models using **ROC-AUC Score**.
- 🏆 Automatically selects the best-performing model.
- 🧹 Performs complete data preprocessing and feature engineering.
- 📊 Interactive Streamlit dashboard with real-time predictions.
- 🎯 Displays churn probability, risk level, and business recommendations.
- 💾 Saves preprocessing objects for consistent future predictions.
- ⚡ Instant prediction using pre-trained models.

---

# 📑 Table of Contents

- [📊 Customer Churn Prediction System](#-customer-churn-prediction-system)
- [🚀 Live Demo](#-live-demo)
- [🌟 Project Highlights](#-project-highlights)
- [📑 Table of Contents](#-table-of-contents)
- [📌 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
  - [🤖 Machine Learning](#-machine-learning)
  - [📊 Streamlit Dashboard](#-streamlit-dashboard)
  - [💼 Business Value](#-business-value)
- [🛠️ Technology Stack](#️-technology-stack)
- [📂 Project Structure](#-project-structure)
- [⚙️ How It Works](#️-how-it-works)
- [📊 Dataset Information](#-dataset-information)
  - [Dataset Summary](#dataset-summary)
  - [Important Features](#important-features)
    - [Customer Information](#customer-information)
    - [Services](#services)
    - [Account Information](#account-information)
    - [Target Variable](#target-variable)
- [🧹 Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
  - [1️⃣ Dataset Loading](#1️⃣-dataset-loading)
  - [2️⃣ Removing Identifier Columns](#2️⃣-removing-identifier-columns)
  - [3️⃣ Missing Value Handling](#3️⃣-missing-value-handling)
    - [Numerical Columns](#numerical-columns)
    - [Categorical Columns](#categorical-columns)
  - [4️⃣ Data Type Conversion](#4️⃣-data-type-conversion)
  - [5️⃣ Target Encoding](#5️⃣-target-encoding)
  - [6️⃣ Categorical Feature Encoding](#6️⃣-categorical-feature-encoding)
  - [7️⃣ Feature Scaling](#7️⃣-feature-scaling)
- [🤖 Machine Learning Models](#-machine-learning-models)
  - [🌲 Random Forest Classifier](#-random-forest-classifier)
    - [Configuration](#configuration)
    - [Advantages](#advantages)
  - [🧠 Artificial Neural Network (ANN)](#-artificial-neural-network-ann)
    - [Model Architecture](#model-architecture)
    - [Training Configuration](#training-configuration)
- [📈 Model Evaluation](#-model-evaluation)
    - [Evaluation Metrics](#evaluation-metrics)
- [🏆 Automatic Best Model Selection](#-automatic-best-model-selection)
- [💾 Saved Model Artifacts](#-saved-model-artifacts)
- [🚀 Installation](#-installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Navigate to the Project Directory](#2-navigate-to-the-project-directory)
  - [3. Install Required Packages](#3-install-required-packages)
- [▶️ Running the Project](#️-running-the-project)
    - [Step 1 — Train the Models](#step-1--train-the-models)
    - [Step 2 — Launch the Streamlit Application](#step-2--launch-the-streamlit-application)
- [✨ Application Features](#-application-features)
  - [🔹 User-Friendly Input Form](#-user-friendly-input-form)
  - [🔹 Intelligent Prediction](#-intelligent-prediction)
  - [🔹 Risk Classification](#-risk-classification)
  - [🔹 Interactive Visualisations](#-interactive-visualisations)
- [🎯 Prediction Output](#-prediction-output)
  - [Example Output](#example-output)
- [📦 Requirements](#-requirements)
    - [Required Libraries](#required-libraries)
- [🚀 Future Improvements](#-future-improvements)
- [📚 Learning Outcomes](#-learning-outcomes)
- [📜 License](#-license)
- [🙏 Acknowledgements](#-acknowledgements)
- [⭐ Support](#-support)
  - [📊 Customer Churn Prediction System](#-customer-churn-prediction-system-1)

---

# 📌 Project Overview

Customer churn prediction is one of the most valuable applications of Machine Learning in customer relationship management. Companies across industries such as telecommunications, banking, insurance, e-commerce, and SaaS use predictive analytics to identify customers who are likely to discontinue their services.

This project implements a complete end-to-end machine learning pipeline that analyses customer information and predicts whether a customer is likely to churn. Rather than relying on a single algorithm, the system trains both a **Random Forest Classifier** and an **Artificial Neural Network (ANN)**, evaluates their performance using **ROC-AUC**, and automatically deploys the better-performing model.

The trained model is integrated into a Streamlit web application where users can enter customer details and receive instant predictions along with churn probability, risk classification, visual indicators, and business-friendly recommendations.

---

# ✨ Key Features

## 🤖 Machine Learning

- Automatic data cleaning and preprocessing.
- Missing value handling.
- Label Encoding for categorical variables.
- StandardScaler for feature normalization.
- Random Forest Classifier implementation.
- Artificial Neural Network using TensorFlow/Keras.
- Automatic model comparison.
- Best model selection based on ROC-AUC score.
- Saved preprocessing pipeline for future predictions.
- Reusable trained models without retraining.

---

## 📊 Streamlit Dashboard

- Interactive customer input form.
- Real-time churn prediction.
- Churn probability calculation.
- Risk level classification.
- Probability indicator.
- Stay vs Churn distribution chart.
- Plain-language business recommendation.
- Automatic loading of trained models.

---

## 💼 Business Value

The application helps organisations:

- Identify customers at high risk of churn.
- Prioritise retention strategies.
- Improve customer satisfaction.
- Reduce customer acquisition costs.
- Support data-driven business decisions.

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Data Visualisation | Matplotlib |
| Web Framework | Streamlit |
| Model Serialization | Joblib |
| Development Tools | VS Code, Jupyter Notebook |

---

# 📂 Project Structure

```text
Customer-Churn-Prediction-System/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── Telco_Customer_Churn.csv
│
├── model_rf.pkl
├── model_ann.keras
├── scaler.pkl
├── encoders.pkl
├── target_encoder.pkl
├── metadata.pkl
│
└── assets/
    └── (Optional project resources)
```

---

# ⚙️ How It Works

The application follows a complete Machine Learning workflow from raw customer data to prediction.

```text
                 Customer Dataset
                        │
                        ▼
               Data Cleaning
                        │
                        ▼
           Feature Engineering
                        │
                        ▼
     Label Encoding + Feature Scaling
                        │
                        ▼
              Train/Test Split
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
 Random Forest                  Neural Network
         │                             │
         └──────────────┬──────────────┘
                        ▼
          Compare ROC-AUC Performance
                        │
                        ▼
         Automatically Select Best Model
                        │
                        ▼
          Save Model & Preprocessing Files
                        │
                        ▼
          Streamlit Web Application
                        │
                        ▼
          Customer Churn Prediction
```

---

# 📊 Dataset Information

This project uses the **Telco Customer Churn Dataset**, one of the most widely used benchmark datasets for binary classification and customer retention analysis.

The dataset contains customer demographic information, subscribed services, billing details, account history, and the customer's churn status.

The objective is to predict whether a customer is likely to discontinue the service based on these features.

---

## Dataset Summary

| Property | Description |
|----------|-------------|
| Dataset | Telco Customer Churn Dataset |
| Problem Type | Binary Classification |
| Domain | Customer Analytics |
| Target Variable | Churn |
| Target Classes | Yes / No |
| Machine Learning Task | Supervised Learning |

---

## Important Features

The dataset contains several customer-related attributes, including:

### Customer Information

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure

### Services

- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Account Information

- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

### Target Variable

```
Churn
```

Meaning:

| Value | Meaning |
|--------|----------|
| Yes | Customer left the company |
| No | Customer stayed with the company |

---

# 🧹 Data Preprocessing Pipeline

Raw data cannot be directly used for machine learning. Several preprocessing steps are performed before training the models.

---

## 1️⃣ Dataset Loading

The dataset is loaded using **Pandas**.

```python
pd.read_csv()
```

---

## 2️⃣ Removing Identifier Columns

The following column is removed:

```
customerID
```

Reason:

- It uniquely identifies customers.
- It contains no predictive information.
- It may negatively affect model performance.

---

## 3️⃣ Missing Value Handling

The dataset is automatically cleaned before training.

### Numerical Columns

Missing values are replaced using the **median** value.

This approach is robust against outliers.

---

### Categorical Columns

Missing values are replaced using the **mode (most frequent value)**.

This preserves the existing category distribution.

---

## 4️⃣ Data Type Conversion

The **TotalCharges** column is converted from text to numeric format.

Any invalid values are safely converted into missing values before being handled during preprocessing.

---

## 5️⃣ Target Encoding

The target variable is transformed into numerical values.

```
No  → 0
Yes → 1
```

This enables the machine learning algorithms to perform binary classification.

---

## 6️⃣ Categorical Feature Encoding

All categorical features are encoded using independent **LabelEncoder** objects.

Each encoder is stored separately and later reused by the Streamlit application to ensure that user inputs are transformed exactly as they were during training.

---

## 7️⃣ Feature Scaling

The project standardises all input features using **StandardScaler**.

Benefits include:

- Improved convergence for neural networks.
- Consistent feature distributions.
- Better numerical stability.
- More reliable predictions.

---

# 🤖 Machine Learning Models

Instead of relying on a single algorithm, this project trains **two independent models** and automatically deploys the better-performing one.

---

## 🌲 Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

### Configuration

| Parameter | Value |
|-----------|-------|
| Estimators | 300 |
| Maximum Depth | 10 |
| Random State | 42 |
| Class Weight | Balanced |

### Advantages

- Excellent performance on structured tabular data.
- Handles complex relationships between features.
- Resistant to overfitting.
- Robust against noisy data.
- Provides strong baseline performance.

---

## 🧠 Artificial Neural Network (ANN)

A fully connected feed-forward neural network is implemented using **TensorFlow/Keras**.

### Model Architecture

```
Input Layer
      │
      ▼
Dense Layer (64 Neurons, ReLU)
      │
Dropout (30%)
      │
Dense Layer (32 Neurons, ReLU)
      │
Dropout (20%)
      │
Dense Layer (16 Neurons, ReLU)
      │
Output Layer (1 Neuron, Sigmoid)
```

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 100 |
| Validation Split | 20% |
| Early Stopping | Enabled |

Early stopping restores the best-performing model weights and prevents unnecessary training once validation performance stops improving.

---

# 📈 Model Evaluation

Both models are evaluated using multiple industry-standard metrics.

### Evaluation Metrics

- Accuracy
- ROC-AUC Score
- Classification Report
- Confusion Matrix

Although accuracy is reported, **ROC-AUC** is used as the primary evaluation metric because it provides a more reliable assessment for binary classification tasks, particularly when class distributions are imbalanced.

---

# 🏆 Automatic Best Model Selection

Once training is complete, both models are compared using their ROC-AUC scores.

```
Random Forest
        │
        ▼
 Compare ROC-AUC
        ▲
        │
Artificial Neural Network
```

The model with the higher ROC-AUC score is automatically selected as the production model.

This design allows the application to always use the strongest available model without requiring any manual changes to the deployment code.

---

# 💾 Saved Model Artifacts

After successful training, all required files are saved for future inference.

| File | Description |
|------|-------------|
| model_rf.pkl | Trained Random Forest model |
| model_ann.keras | Trained Artificial Neural Network |
| scaler.pkl | StandardScaler object |
| encoders.pkl | Label Encoders for categorical features |
| target_encoder.pkl | Target Label Encoder |
| metadata.pkl | Feature names, ranges, categories, and model information |

These files are automatically loaded by the Streamlit application, eliminating the need to retrain the models each time the application starts.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/govindchaturvedi1995-svg/Customer-Churn-Prediction-Model
```

---

## 2. Navigate to the Project Directory

```bash
cd Customer-Churn-Prediction-System
```

---

## 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Step 1 — Train the Models

Execute:

```bash
python train_model.py
```

This script will:

- Load the dataset.
- Clean and preprocess the data.
- Train both machine learning models.
- Compare their performance.
- Save the trained models and preprocessing objects.

---

### Step 2 — Launch the Streamlit Application

Run:

```bash
streamlit run app.py
```

Once the Streamlit server starts, open the displayed local URL in your browser to access the application.

---


# ✨ Application Features

The Streamlit application provides a simple and interactive interface for making customer churn predictions in real time.

## 🔹 User-Friendly Input Form

Users can enter customer information through an intuitive form without requiring any technical knowledge.

Supported input fields include:

- Customer Demographics
- Service Details
- Contract Information
- Billing Information
- Payment Method
- Account Details

---

## 🔹 Intelligent Prediction

After entering the customer details, the application automatically:

- Preprocesses the input data.
- Encodes categorical features.
- Scales numerical values.
- Loads the best-performing trained model.
- Predicts customer churn probability.

---

## 🔹 Risk Classification

The predicted probability is converted into an easy-to-understand business risk level.

| Churn Probability | Risk Level |
|------------------|------------|
| 0% – 29% | 🟢 Low Risk |
| 30% – 49% | 🟡 Moderate Risk |
| 50% – 69% | 🟠 High Risk |
| 70% – 100% | 🔴 Very High Risk |

---

## 🔹 Interactive Visualisations

The application presents prediction results using multiple visual components, including:

- Churn Probability Indicator
- Probability Progress Bar
- Stay vs Churn Pie Chart
- Business-Friendly Summary

These visualisations make the prediction easier to interpret for both technical and non-technical users.

---

# 🎯 Prediction Output

After prediction, the application displays:

- Customer Status (Likely to Stay / Likely to Churn)
- Churn Probability (%)
- Risk Category
- Visual Probability Indicator
- Stay vs Churn Distribution
- Business Recommendation

---

## Example Output

```text
Prediction Status

Likely to Churn

------------------------------------

Churn Probability

82.4%

------------------------------------

Risk Level

Very High Risk

------------------------------------

Recommendation

Consider proactive customer retention
through personalised offers,
customer support, or loyalty incentives.
```

---

# 📦 Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

### Required Libraries

- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

# 🚀 Future Improvements

Possible future enhancements for this project include:

- Implementation of XGBoost and LightGBM models.
- Hyperparameter optimisation using Grid Search or Optuna.
- Model explainability using SHAP or LIME.
- Feature importance visualisation.
- Batch prediction using CSV uploads.
- Downloadable prediction reports.
- Email-based prediction reports.
- REST API integration using FastAPI.
- Docker containerisation.
- Cloud deployment with CI/CD.
- Database integration for storing prediction history.
- User authentication and role-based access.
- Automated model retraining using new datasets.

---

# 📚 Learning Outcomes

This project demonstrates practical implementation of:

- Data Cleaning
- Data Preprocessing
- Feature Engineering
- Feature Scaling
- Label Encoding
- Binary Classification
- Random Forest
- Artificial Neural Networks
- Model Evaluation
- Model Selection
- Model Serialization
- Machine Learning Deployment
- Streamlit Application Development

---

# 📜 License

This project is licensed under the **MIT License**.

See the **LICENSE** file for additional details.

---

# 🙏 Acknowledgements

Special thanks to the developers and maintainers of the following open-source technologies that made this project possible:

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow
- Streamlit
- Matplotlib
- Joblib

Their contributions to the open-source community continue to make machine learning and data science more accessible to developers around the world.

---

# ⭐ Support

If you found this project useful, consider supporting it by:

- ⭐ Starring this repository
- 🍴 Forking the repository
- 🛠️ Suggesting improvements
- 🐞 Reporting bugs
- 📢 Sharing it with others

Your support is greatly appreciated.

---

<div align="center">

## 📊 Customer Churn Prediction System

**Predict • Analyse • Retain**

Built with ❤️ using **Python**, **Scikit-learn**, **TensorFlow**, and **Streamlit**.

⭐ If you like this project, don't forget to star the repository!

</div>
