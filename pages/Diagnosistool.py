# pages/1_Diagnosis_Tool.py

import streamlit as st
from PIL import Image
import numpy as np
import time

# --- DUMMY AI FUNCTION ---
# This function simulates the AI model's output.
# The AI Support person will replace this with the real model later.

def get_dummy_prediction(image_array):
    """
    A placeholder function that mimics the model's output.
    It returns a dictionary with prediction details.
    """
    # Simulate processing time
    with st.spinner('AI is analyzing the image...'):
        time.sleep(2) # Simulate a 2-second delay

    # In a real scenario, the model would return these values.
    # We are hardcoding them here for the UI demo.
    return {
        "stage": 2,
        "stage_name": "Moderate Non-proliferative DR",
        "confidence": 0.88,
        "heatmap": Image.open("placeholder_heatmap.png") # A placeholder image
    }

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diagnosis Tool", page_icon="🩺", layout="wide")


# --- HELPER FUNCTION to create a placeholder heatmap ---
# This is just for the demo. The real heatmap will come from Grad-CAM.
def create_placeholder_image(size=(512, 512), text="Placeholder"):
    img = Image.new('RGB', size, color = 'grey')
    return img

# Create a placeholder heatmap if it doesn't exist
try:
    placeholder_heatmap = Image.open("placeholder_heatmap.png")
except FileNotFoundError:
    placeholder_heatmap = create_placeholder_image(text="Grad-CAM Heatmap")
    placeholder_heatmap.save("placeholder_heatmap.png")


# --- UI LAYOUT ---
st.title("🩺 Diabetic Retinopathy Diagnosis Interface")
st.markdown(
    "Upload a retinal fundus image to get an AI-powered diagnosis and explanation."
)

# ICDR Scale Information Card in the sidebar
st.sidebar.title("ICDR Severity Scale")
st.sidebar.info(
    """
    - **0:** No DR
    - **1:** Mild Non-proliferative DR
    - **2:** Moderate Non-proliferative DR
    - **3:** Severe Non-proliferative DR
    - **4:** Proliferative DR
    """
)


# Main layout with two columns for image comparison
col1, col2 = st.columns(2)

with col1:
    st.header("Image for Analysis")
    uploaded_file = st.file_uploader(
        "Choose a retinal image...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

        # "Diagnose" button
        if st.button("Diagnose Image", use_container_width=True, type="primary"):
            # Call the dummy prediction function
            prediction = get_dummy_prediction(np.array(image))

            # --- DISPLAY PREDICTION RESULTS ---
            st.subheader("🤖 AI Diagnosis Results")

            # Display predicted stage and confidence
            stage_name = prediction['stage_name']
            confidence = prediction['confidence']
            st.metric(label="Predicted Severity", value=stage_name, delta=f"Confidence: {confidence:.2%}")

            # Display the Grad-CAM heatmap
            st.subheader("Explainability: What the AI is looking at")
            st.image(
                prediction['heatmap'],
                caption="Grad-CAM Heatmap overlay shows where the model focused.",
                use_column_width=True
            )

with col2:
    st.header("Comparison Image")
    comparison_file = st.file_uploader(
        "Upload a reference image (e.g., a healthy retina)...", type=["jpg", "png", "jpeg"]
    )

    if comparison_file is not None:
        comp_image = Image.open(comparison_file)
        st.image(comp_image, caption="Reference Image", use_column_width=True)
    else:
        st.info("Upload a second image here to compare side-by-side.")