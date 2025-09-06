# pages/1_Diagnosis_Tool.py

import streamlit as st
from PIL import Image
import numpy as np
import time
import pandas as pd

# --- DUMMY AI FUNCTION ---
# This simulates the AI model's output and is now more detailed.
def get_dummy_prediction(image_array):
    """
    A placeholder function that mimics the model's output.
    It now randomly selects a stage and returns a full set of fake
    confidence scores for the bar chart.
    """
    with st.spinner('AI is analyzing the image...'):
        time.sleep(2)  # Simulate a 2-second delay

    import random
    
    # Generate random confidences for each stage
    confidences = [random.uniform(0.1, 0.9) for _ in range(5)]
    total_confidence = sum(confidences)
    
    # Normalize the confidences to sum to 1.0 (for a better chart)
    normalized_confidences = [c / total_confidence for c in confidences]

    # Find the predicted stage (highest confidence)
    predicted_stage_index = normalized_confidences.index(max(normalized_confidences))

    return {
        "stage": predicted_stage_index,
        "stage_name": ICDR_SCALE_INFO[predicted_stage_index]['name'],
        "confidence": normalized_confidences[predicted_stage_index],
        "all_confidences": {
            "Stage": [info["name"] for info in ICDR_SCALE_INFO.values()],
            "Confidence": normalized_confidences
        },
        "heatmap": create_placeholder_image(text="Grad-CAM Heatmap")
    }

# --- HELPER FUNCTIONS ---
def create_placeholder_image(size=(512, 512), text="Placeholder"):
    img = Image.new('RGB', size, color='grey')
    return img

# --- UI MARKDOWN STYLES ---
st.markdown("""
<style>
.severity-box {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    text-align: center;
    transition: all 0.3s ease;
}
.severity-box.severity-0 { background-color: #A5D6A7; border-color: #81C784; } /* Brighter Green */
.severity-box.severity-1 { background-color: #A5D6A7; border-color: #81C784; } /* Brighter Green */
.severity-box.severity-2 { background-color: #FFF176; border-color: #FFEB3B; } /* Brighter Yellow */
.severity-box.severity-3 { background-color: #EF9A9A; border-color: #E57373; } /* Brighter Red */
.severity-box.severity-4 { background-color: #EF9A9A; border-color: #E57373; } /* Brighter Red */

.stButton>button {
    height: 3em;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# --- ICDR SCALE & LINKS DATA ---
ICDR_SCALE_INFO = {
    0: {"name": "No DR", "color_class": "severity-0"},
    1: {"name": "Mild Non-proliferative DR", "color_class": "severity-1"},
    2: {"name": "Moderate Non-proliferative DR", "color_class": "severity-2"},
    3: {"name": "Severe Non-proliferative DR", "color_class": "severity-3"},
    4: {"name": "Proliferative DR", "color_class": "severity-4"}
}

RELATED_LINKS = {
    0: [
        {"text": "Diabetic Eye Disease Information", "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy"},
        {"text": "Preventing Diabetic Retinopathy", "url": "https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy"}
    ],
    1: [
        {"text": "About Mild Non-proliferative DR", "url": "https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy"},
        {"text": "Managing Early Stage DR", "url": "https://www.webmd.com/diabetes/diabetic-retinopathy-stages"}
    ],
    2: [
        {"text": "About Moderate Non-proliferative DR", "url": "https://www.eyecarecolorado.com/diabetic-retinopathy-stages-denver/"},
        {"text": "Treatments for Diabetic Retinopathy", "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy/treating-diabetic-retinopathy"}
    ],
    3: [
        {"text": "About Severe Non-proliferative DR", "url": "https://www.webmd.com/diabetes/diabetic-retinopathy-stages"},
        {"text": "Understanding Advanced DR", "url": "https://www.healthline.com/health/diabetes/diabetic-retinopathy-stages"}
    ],
    4: [
        {"text": "About Proliferative DR (PDR)", "url": "https://www.moorfields.nhs.uk/mediaLocal/uvqpwxrq/proliferative-diabetic-retinopathy-pdr.pdf"},
        {"text": "Advanced Treatments for PDR", "url": "https://www.bayarearetina.com/proliferative-diabetic-retinopathy"}
    ]
}

# --- PAGE LAYOUT ---
st.title("🩺 Diabetic Retinopathy Diagnosis Interface")
st.markdown("Upload a retinal fundus image to get an AI-powered diagnosis and explanation.")

# ICDR Scale at the top of the page
st.subheader("ICDR Severity Scale")
st.markdown("The AI model classifies images into these stages based on the following scale:")
cols = st.columns(5)
for stage, info in ICDR_SCALE_INFO.items():
    with cols[stage]:
        st.markdown(
            f'<div class="severity-box {info["color_class"]}">'
            f'<b>Stage {stage}</b><br>{info["name"]}'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("---")

# Initialize session state to store the prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Single image upload and diagnosis
st.header("Image for Analysis")
uploaded_file = st.file_uploader(
    "Choose a retinal image...", type=["jpg", "png", "jpeg"], key="main_uploader"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    if st.button("Diagnose Image", use_container_width=True, type="primary"):
        st.session_state.prediction = get_dummy_prediction(np.array(image))
        # Rerun to update the UI with the result
        st.experimental_rerun()

# --- DISPLAY RESULTS (if a prediction has been made) ---
if st.session_state.prediction:
    pred = st.session_state.prediction
    
    st.markdown("---")
    st.subheader("🤖 AI Diagnosis Results")

    # Bar chart showing confidence for all stages
    st.markdown("##### Model Confidence by Stage")
    df = pd.DataFrame(pred['all_confidences'])
    df['Confidence'] = df['Confidence'].map('{:.2%}'.format)
    st.bar_chart(df, x="Stage", y="Confidence")

    # Text box for the main diagnosis
    severity_class = ICDR_SCALE_INFO[pred['stage']]['color_class']
    st.markdown(
        f'<div class="severity-box {severity_class}">'
        f'<h3>{pred["stage_name"]}</h3>'
        f'Confidence: <b>{pred["confidence"]:.2%}</b>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.subheader("Explainability: What the AI is looking at")
    st.image(
        pred['heatmap'],
        caption="A Grad-CAM Heatmap shows the areas the model focused on to make its diagnosis. Warmer colors indicate higher importance.",
        use_column_width=True
    ) 

    # --- DISPLAY RELATED LINKS ---
    st.subheader("📚 Learn More About This Stage")
    links = RELATED_LINKS.get(pred['stage'], [])
    for link in links:
        st.markdown(f"- [{link['text']}]({link['url']})", unsafe_allow_html=True)