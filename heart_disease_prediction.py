# Heart Disease Prediction
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("heart_disease.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

# Check missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Clean missing numerical values using median
numeric_columns = data.select_dtypes(include=["number"]).columns
for column in numeric_columns:
    if data[column].isnull().any():
        data[column] = data[column].fillna(data[column].median())

print("\nMissing values after cleaning:", data.isnull().sum().sum())

# Features and target
X = data.drop(columns=["heart_disease"])
y = data["heart_disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classification model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict test data
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No Heart Disease", "Heart Disease"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new example
new_person = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "resting_bp": 140,
    "cholesterol": 240,
    "fasting_blood_sugar": 0,
    "max_heart_rate": 150,
    "chest_pain_type": 1,
    "oldpeak": 1.0,
    "resting_ecg": 1,
    "exercise_angina": 0,
    "st_slope": 1
}])

new_person_scaled = scaler.transform(new_person)
prediction = model.predict(new_person_scaled)[0]

print("\nExample prediction:",
      "Heart Disease" if prediction == 1 else "No Heart Disease")

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(
    data["age"],
    data["cholesterol"],
    c=data["heart_disease"],
    alpha=0.7
)
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("Heart Disease Feature Analysis")
plt.tight_layout()
plt.savefig("heart_disease_analysis.png", dpi=150)
plt.show()
