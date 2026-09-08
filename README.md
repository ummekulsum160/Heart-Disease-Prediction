# Heart-Disease-Prediction
Heart Disease Prediction using Machine Learning is a machine-learning project that predicts whether a person may have heart disease based on medical and health-related information such as age, sex, resting blood pressure, cholesterol, maximum heart rate, chest pain type, exercise-induced angina, and other clinical features.
# Heart Disease Prediction

A Machine Learning project that predicts whether a person is likely to have heart disease based on various health and clinical features.

The project uses Python, Pandas, Matplotlib, and Scikit-learn. Logistic Regression is used as the classification algorithm, along with data preprocessing, feature scaling, model training, prediction, and performance evaluation.

## Features

* Data loading and exploration
* Missing-value handling
* Data preprocessing
* Train-test splitting
* Feature scaling using StandardScaler
* Heart disease prediction using Logistic Regression
* Model evaluation using accuracy, classification report, and confusion matrix
* Data visualization using Matplotlib

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn

## Dataset

The dataset contains health-related features such as age, sex, blood pressure, cholesterol, maximum heart rate, chest pain type, and other clinical attributes.

### Target

* `0` - No Heart Disease
* `1` - Heart Disease

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd heart-disease-prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**

```bash
venv\Scripts\activate
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Python Program

```bash
python heart_disease_prediction.py
```

### 2. View the Output

The program performs data preprocessing, trains the Logistic Regression model, and displays:

* Dataset information
* Missing values
* Model accuracy
* Classification report
* Confusion matrix
* Sample predictions

The program also generates a visualization for analyzing the relationship between selected health features.

## Project Structure

```text
Heart_Disease_Prediction/
│
├── heart_disease.csv
├── heart_disease_prediction.py
├── requirements.txt
├── README.md
└── heart_disease_analysis.png
```

## Disclaimer

This project is created for educational purposes only and should not be considered a medical diagnostic system.
