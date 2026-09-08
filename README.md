# Heart Disease Prediction

## Synopsis
This project uses machine learning to predict whether a person may have heart disease based on health-related features such as age, blood pressure, cholesterol, heart rate, chest pain, and other clinical measurements.

## Objectives
- Explore the heart disease dataset.
- Analyze important health-related features.
- Prepare and clean data for machine learning.
- Train a classification model.
- Predict the possibility of heart disease.
- Evaluate prediction results.

## Tools
Python, Pandas, Matplotlib, Scikit-learn

## Dataset Features
- age
- sex
- resting_bp
- cholesterol
- fasting_blood_sugar
- max_heart_rate
- chest_pain_type
- oldpeak
- resting_ecg
- exercise_angina
- st_slope
- heart_disease

Target:
- 0 = No Heart Disease
- 1 = Heart Disease

## Data Preparation
- Loaded the CSV dataset with Pandas.
- Checked for missing values.
- Filled missing numerical values with their median.
- Split the data into training and testing sets.
- Standardized numerical features using StandardScaler.

## Machine Learning Algorithm
Logistic Regression is used for binary classification.

## Evaluation
The program calculates:
- Accuracy
- Classification report
- Confusion matrix

It also creates a scatter plot showing the relationship between age and cholesterol.

## How to Run
1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   python heart_disease_prediction.py

## Important Note
This is an educational machine-learning project using a classroom dataset. It is not a medical diagnostic tool and should not be used for clinical decisions.
