import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Prediction")
st.write("Enter the patient's information to get a model prediction.")

@st.cache_data
def load_data():
    data = pd.read_csv("heart_disease.csv")

    numeric_columns = data.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if data[column].isnull().any():
            data[column] = data[column].fillna(data[column].median())

    return data

@st.cache_resource
def train_model(data):
    X = data.drop(columns=["heart_disease"])
    y = data["heart_disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist()

try:
    data = load_data()

    if "heart_disease" not in data.columns:
        st.error("The dataset must contain a 'heart_disease' column.")
        st.stop()

    model, scaler, feature_columns = train_model(data)

except FileNotFoundError:
    st.error(
        "heart_disease.csv was not found. "
        "Keep app.py and heart_disease.csv in the same folder."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading or training the model: {e}")
    st.stop()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55, step=1)

    sex = st.selectbox(
        "Sex (0 = Female, 1 = Male)",
        options=[0, 1],
        index=1
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=50, max_value=250, value=140, step=1
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=50, max_value=700, value=240, step=1
    )

    fasting_blood_sugar = st.selectbox(
        "Fasting Blood Sugar (0 = Normal, 1 = High)",
        options=[0, 1],
        index=0
    )

    max_heart_rate = st.number_input(
        "Maximum Heart Rate",
        min_value=40, max_value=250, value=150, step=1
    )

with col2:
    chest_pain_type = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        index=1
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0, max_value=15.0, value=1.0, step=0.1
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2],
        index=1
    )

    exercise_angina = st.selectbox(
        "Exercise Angina (0 = No, 1 = Yes)",
        options=[0, 1],
        index=0
    )

    st_slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2],
        index=1
    )

if st.button("Predict Heart Disease", use_container_width=True):

    new_person = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "max_heart_rate": max_heart_rate,
        "chest_pain_type": chest_pain_type,
        "oldpeak": oldpeak,
        "resting_ecg": resting_ecg,
        "exercise_angina": exercise_angina,
        "st_slope": st_slope
    }])

    try:
        new_person = new_person[feature_columns]
    except KeyError:
        st.error("The input fields do not match the dataset columns.")
        st.write("Dataset features found:", feature_columns)
        st.stop()

    new_person_scaled = scaler.transform(new_person)
    prediction = model.predict(new_person_scaled)[0]
    probability = model.predict_proba(new_person_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    st.write(
        f"Model-estimated probability of heart disease: "
        f"**{probability * 100:.2f}%**"
    )

    st.caption(
        "This result is for educational purposes only and is not a medical diagnosis."
    )

with st.expander("About this project"):
    st.write(
        "This app uses the same machine-learning pipeline as the original "
        "project: missing-value cleaning, train-test split, StandardScaler, "
        "and Logistic Regression."
    )
    st.write(
        f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns"
    )
    st.write("Features used by the model:")
    st.write(feature_columns)
