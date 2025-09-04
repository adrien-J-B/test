import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle

st.set_page_config(page_title="Risk Predictor", page_icon="⚠️")

st.title("Student Burnout Risk Predictor ⚠️")
st.write("Enter student data below or upload a CSV for batch prediction.")

# Define the model architecture (must match training)
class BurnoutPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

# --- Load model and scaler ---
@st.cache_resource
def load_model():
    try:
        # Load model info
        with open("model_info.pkl", "rb") as f:
            model_info = pickle.load(f)
        
        # Create model with correct architecture
        model = BurnoutPredictor(model_info['input_size'])
        
        # Load state dict (safer than loading entire model)
        model.load_state_dict(torch.load("model_state_dict.pt", map_location=torch.device('cpu')))
        model.eval()  # Set to evaluation mode
        
        # Load scaler
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            
        return model, scaler, model_info
        
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.error("Please run create_dummy_model.py first to generate the model files.")
        st.stop()

try:
    model, scaler, model_info = load_model()
    faculty_options = model_info['faculty_options']
    feature_cols = model_info['feature_cols']
except:
    st.error("Model files not found. Please run create_dummy_model.py first.")
    st.stop()

st.subheader("Single Student Check")
with st.form("single_student_form"):
    attendance_rate = st.slider("Attendance Rate", 0.0, 1.0, 0.8)
    avg_grade = st.number_input("Average Grade", 0, 100, 70)
    hours_studied = st.number_input("Hours Studied per Week", 0, 40, 5)
    sleep_hours = st.number_input("Sleep Hours per Night", 0, 12, 7)
    exercise_per_week = st.number_input("Exercise Sessions per Week", 0, 7, 2)
    faculty = st.selectbox("Faculty", faculty_options)
    submit_single = st.form_submit_button("Predict Risk")

if submit_single:
    # Create input DataFrame
    input_df = pd.DataFrame([{
        "attendance_rate": attendance_rate,
        "avg_grade": avg_grade,
        "hours_studied_per_week": hours_studied,
        "sleep_hours": sleep_hours,
        "exercise_per_week": exercise_per_week,
        "faculty": faculty
    }])
    
    # One-hot encode faculty
    input_df = pd.get_dummies(input_df, columns=["faculty"], drop_first=False)
    
    # Ensure all faculty columns are present
    for fac in faculty_options:
        col = f"faculty_{fac}"
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training data
    input_df = input_df[feature_cols]
    
    # Scale the input
    X_scaled = scaler.transform(input_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(X_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    
    # Display results
    st.markdown(f"### Prediction: {'🟠 At Risk' if pred else '🟢 Safe'}")
    st.progress(prob)
    st.write(f"Probability of burnout risk: **{prob:.2%}**")
    
    # Add some interpretation
    if pred:
        st.warning("This student shows signs of potential burnout. Consider recommending counseling services.")
    else:
        st.success("This student appears to be managing well. Continue current support practices.")

st.subheader("Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
    
    # Create a copy for processing
    df_clean = df.copy()
    
    # Handle missing values for numerical columns
    for col in ["attendance_rate", "avg_grade", "hours_studied_per_week", "sleep_hours", "exercise_per_week"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            st.error(f"Missing required column: {col}")
            st.stop()
    
    # Handle faculty column
    if "faculty" in df_clean.columns:
        df_clean["faculty"] = df_clean["faculty"].fillna(df_clean["faculty"].mode()[0] if not df_clean["faculty"].mode().empty else "Science")
        # Ensure faculty values are valid
        df_clean["faculty"] = df_clean["faculty"].apply(lambda x: x if x in faculty_options else "Science")
    else:
        df_clean["faculty"] = "Science"
    
    # One-hot encode faculty
    df_clean = pd.get_dummies(df_clean, columns=["faculty"], drop_first=False)
    
    # Ensure all faculty columns are present
    for fac in faculty_options:
        col = f"faculty_{fac}"
        if col not in df_clean.columns:
            df_clean[col] = 0
    
    # Ensure we have all required columns
    for col in feature_cols:
        if col not in df_clean.columns:
            st.error(f"Processed data missing column: {col}")
            st.stop()
    
    # Select only the required columns in the right order
    df_clean = df_clean[feature_cols]
    
    # Scale the data
    X_scaled = scaler.transform(df_clean)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.sigmoid(outputs).numpy().flatten()
        preds = (probs > 0.5).astype(int)
    
    # Create results DataFrame
    result_df = df.copy()
    result_df["risk_prediction"] = preds
    result_df["risk_probability"] = probs
    result_df["risk_status"] = result_df["risk_prediction"].apply(lambda x: "At Risk" if x == 1 else "Safe")
    
    st.write("Batch prediction results:")
    st.dataframe(result_df)
    
    # Download button
    csv = result_df.to_csv(index=False)
    st.download_button(
        "Download Results as CSV",
        csv,
        "risk_predictions.csv",
        "text/csv"
    )
    
    # Summary chart
    st.subheader("Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        at_risk_count = result_df["risk_prediction"].sum()
        total_count = len(result_df)
        st.metric("At Risk Students", f"{at_risk_count}/{total_count}", f"{at_risk_count/total_count:.1%}")
        
    with col2:
        avg_prob = result_df["risk_probability"].mean()
        st.metric("Average Risk Probability", f"{avg_prob:.1%}")
    
    st.bar_chart(result_df["risk_status"].value_counts())

# Add a sample download
st.subheader("Sample Data")
st.write("Download a sample CSV file to test the batch prediction feature:")
sample_df = pd.DataFrame({
    "attendance_rate": [0.92, 0.65, 0.88],
    "avg_grade": [72, 48, 69],
    "hours_studied_per_week": [6, 2, 5],
    "sleep_hours": [7, 4, 6],
    "exercise_per_week": [2, 0, 2],
    "faculty": ["Science", "Arts", "Business"]
})
st.dataframe(sample_df)

sample_csv = sample_df.to_csv(index=False)
st.download_button(
    "Download Sample CSV",
    sample_csv,
    "sample_student_data.csv",
    "text/csv"
)

st.info("Tip: For best results, use the sample CSV format shown above. Missing values and unknown faculties are handled automatically.")