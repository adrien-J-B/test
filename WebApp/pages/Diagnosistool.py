# pages/Diagnosistool.py

import streamlit as st
from PIL import Image
import numpy as np
import time
import pandas as pd
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="DR-AI-Vision | Diagnosis Tool",
    page_icon="favicon1.png",
    layout="wide",
)

# --- DUMMY AI FUNCTION ---
def get_dummy_prediction(image_array, risk_factors=None):
    with st.spinner('AI is analyzing the image...'):
        time.sleep(2)
    
    # Base confidence values
    confidences = [random.uniform(0.1, 0.9) for _ in range(5)]
    total_confidence = sum(confidences)
    normalized_confidences = [c / total_confidence for c in confidences]

    # Adjust confidences based on risk factors
    if risk_factors:
        risk_score = 0
        # Add points for each risk factor, with higher-impact factors getting more points
        if risk_factors.get("diabetes_duration", 0) > 10:
            risk_score += 0.1
        if risk_factors.get("blood_sugar_control", 2) < 2: # 0=Poor, 1=Fair
            risk_score += 0.2
        if risk_factors.get("tobacco_use", False):
            risk_score += 0.15
        if risk_factors.get("pregnancy", False):
            risk_score += 0.2
        if risk_factors.get("high_bp", False):
            risk_score += 0.1
        if risk_factors.get("high_cholesterol", False):
            risk_score += 0.1
        
        # Shift probability towards higher stages based on the calculated risk score
        if risk_score > 0:
            for i in range(len(normalized_confidences)):
                # Decrease confidence for lower stages (0 and 1)
                if i < 2:
                    normalized_confidences[i] = max(0, normalized_confidences[i] - risk_score/2)
                # Increase confidence for higher stages (3 and 4)
                if i > 2:
                    normalized_confidences[i] = min(1, normalized_confidences[i] + risk_score)
            
            # Re-normalize the confidences to sum to 1
            total_confidence_new = sum(normalized_confidences)
            normalized_confidences = [c / total_confidence_new for c in normalized_confidences]

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
    img = Image.new('RGB', size, color='#cccccc')
    return img

# --- Custom CSS to mimic C-Care ---
st.markdown("""
<style>
    /* Global Fonts and Colors */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
        color: #333333;
        background-color: #f6f7f9;
    }
    
    /* Main container and section styling */
    .st-emotion-cache-183-b7-e2 {
        padding: 0;
    }
    .st-emotion-cache-183-b7-e2 .st-emotion-cache-183-b7-e2 {
        padding: 2rem 4rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #004d99;
        font-weight: 500;
    }
    
    .section-title {
        font-size: 2rem;
        font-weight: 500;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .instruction-text {
        text-align: center;
        margin-bottom: 2rem;
        color: #666666;
    }
    
    /* ICDR Scale Styling */
    .icdr-scale-container {
        display: flex;
        justify-content: space-around;
        gap: 1.5rem;
        margin-bottom: 3rem;
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .icdr-scale-box {
        text-align: center;
        flex: 1;
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .icdr-scale-box:hover {
        transform: translateY(-5px);
    }
    .icdr-scale-box h5 {
        margin-bottom: 0.5rem;
        color: #ffffff;
        font-weight: 600;
    }
    .icdr-scale-box p {
        margin: 0;
        font-size: 0.9em;
        color: #f0f0f0;
    }
    
    /* Specific colors for each stage */
    .icdr-scale-box.stage-0 { background-color: #5cb85c; border-color: #4cae4c; }
    .icdr-scale-box.stage-1 { background-color: #8cd47e; border-color: #7bbd6c; }
    .icdr-scale-box.stage-2 { background-color: #f0ad4e; border-color: #eea236; }
    .icdr-scale-box.stage-3 { background-color: #d9534f; border-color: #d43f3a; }
    .icdr-scale-box.stage-4 { background-color: #c9302c; border-color: #ac2925; }

    /* Main Content Blocks */
    .content-block {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #008080 !important; /* Deep sea-green */
    }
    
    /* Button Styling */
    .st-emotion-cache-183-b7-e2 button {
        background-color: #004d99;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 1rem 2rem;
        font-size: 1rem;
        font-weight: 500;
        transition: background-color 0.3s, transform 0.2s;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-183-b7-e2 button:hover {
        background-color: #003366;
        transform: translateY(-2px);
    }
    
    /* Prediction Result Box */
    .prediction-box {
        background-color: #e6f7ff;
        border-left: 5px solid #004d99;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .prediction-box h4 {
        color: #004d99;
        margin-top: 0;
    }
    .prediction-box b {
        color: #333333;
    }
    
    /* More specific CSS for the links */
    .content-block li a {
        font-size: 1.2em !important;
        line-height: 1.8 !important;
        margin-bottom: 0.5em !important;
        display: block !important;
    }

    /* Style for the resized image preview */
    .stImage > img {
        max-width: 50% !important;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Adjust button placement */
    .center-button {
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# --- ICDR SCALE & LINKS DATA ---
ICDR_SCALE_INFO = {
    0: {"name": "No DR"},
    1: {"name": "Mild NPDR"},
    2: {"name": "Moderate NPDR"},
    3: {"name": "Severe NPDR"},
    4: {"name": "Proliferative DR"},
}

RELATED_LINKS = {
    0: [
        {"text": "Diabetic Eye Disease Information", "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy"},
        {"text": "American Academy of Ophthalmology: DR", "url": "https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy"},
        {"text": "Diabetes.org: Eye Complications", "url": "https://www.diabetes.org/diabetes/complications/eye-complications"},
    ],
    1: [
        {"text": "Managing Early Stage DR", "url": "https://www.webmd.com/diabetes/diabetic-retinopathy-stages"},
        {"text": "Preventing DR Progression", "url": "https://www.cdc.gov/diabetes/managing/complications.html"},
        {"text": "Nonproliferative Diabetic Retinopathy", "url": "https://www.retina.org/resources/patient-information/macular-disease/diabetic-retinopathy"},
    ],
    2: [
        {"text": "Treatments for Diabetic Retinopathy", "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy/treating-diabetic-retinopathy"},
        {"text": "Retina Associates of New York: DR", "url": "https://www.retinany.com/what-is-diabetic-retinopathy/"},
        {"text": "Eye Health Information on DR", "url": "https://www.healthline.com/health/diabetes/diabetic-retinopathy-stages"},
    ],
    3: [
        {"text": "Understanding Advanced DR", "url": "https://www.healthline.com/health/diabetes/diabetic-retinopathy-stages"},
        {"text": "Severe Nonproliferative DR Overview", "url": "https://www.aao.org/eye-health/diseases/diabetic-retinopathy-symptoms-treatment"},
        {"text": "Clinical Practice Guidelines for DR", "url": "https://care.diabetesjournals.org/content/36/suppl_1/S19"},
    ],
    4: [
        {"text": "Advanced Treatments for PDR", "url": "https://www.bayarearetina.com/proliferative-diabetic-retinopathy"},
        {"text": "Proliferative Diabetic Retinopathy (PDR)", "url": "https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy/treating-diabetic-retinopathy"},
        {"text": "Vitrectomy and other PDR Treatments", "url": "https://www.moorfields.nhs.uk/mediaLocal/uvqpwxrq/proliferative-diabetic-retinopathy-pdr.pdf"},
    ]
}

# --- PAGE LAYOUT ---
st.markdown("<h2 class='section-title'>Diabetic Retinopathy Diagnosis Interface</h2>", unsafe_allow_html=True)
st.markdown("<p class='instruction-text'>Upload a retinal image to receive an AI-powered diagnosis.</p>", unsafe_allow_html=True)

# ICDR Scale Section
st.subheader("Classification Scale")
st.markdown("<div class='icdr-scale-container'>", unsafe_allow_html=True)
cols = st.columns(5)
for stage, info in ICDR_SCALE_INFO.items():
    with cols[stage]:
        st.markdown(
            f'<div class="icdr-scale-box stage-{stage}">'
            f'<h5>Stage {stage}</h5>'
            f'<p>{info["name"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
st.markdown("</div>", unsafe_allow_html=True)

# Main Diagnosis Block
with st.container():
    st.markdown("<div class='content-block'>", unsafe_allow_html=True)
    st.subheader("Step 1: Upload Patient's Retinal Image")

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    uploaded_file = st.file_uploader(
        "Choose a retinal image...", type=["jpg", "png", "jpeg"], key="main_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display the uploaded image and Run Diagnosis button
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
        with col2:
            st.markdown("### Initial Analysis")
            st.markdown("Click the button below to get an initial diagnosis based on the fundus image alone.")
            if st.button("Run AI Diagnosis", use_container_width=True):
                st.session_state.prediction = get_dummy_prediction(np.array(image))
                st.session_state.image_uploaded = True
                st.rerun()

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- NEW: RISK FACTOR FORM ---
        with st.form("risk_factor_form"):
            st.subheader("Step 2: Add Risk Factor Details (Optional)")
            st.markdown("Input additional information to refine the probability of a higher-risk diagnosis.")
            
            # Form Inputs
            col_form1, col_form2 = st.columns(2)
            with col_form1:
                diabetes_duration = st.number_input(
                    "Diabetes Duration (in years)", min_value=0, max_value=50, value=5, step=1
                )
                blood_sugar_control = st.radio(
                    "Blood Sugar Control",
                    options=["Poor", "Fair", "Good"],
                    index=2, # Default to Good
                )
            
            with col_form2:
                is_pregnant = st.checkbox("Currently Pregnant")
                uses_tobacco = st.checkbox("Tobacco User")
                has_high_bp = st.checkbox("High Blood Pressure")
                has_high_cholesterol = st.checkbox("High Cholesterol")

            submitted = st.form_submit_button("Recalculate with Risk Factors")
            
            if submitted:
                risk_factors = {
                    "diabetes_duration": diabetes_duration,
                    "blood_sugar_control": ["Poor", "Fair", "Good"].index(blood_sugar_control),
                    "pregnancy": is_pregnant,
                    "tobacco_use": uses_tobacco,
                    "high_bp": has_high_bp,
                    "high_cholesterol": has_high_cholesterol
                }
                # Recalculate and update the session state
                st.session_state.prediction = get_dummy_prediction(np.array(image), risk_factors)
                st.rerun()
                
    st.markdown("</div>", unsafe_allow_html=True)

# --- DISPLAY RESULTS (if a prediction has been made) ---
if st.session_state.prediction:
    pred = st.session_state.prediction
    
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='content-block'>", unsafe_allow_html=True)
        st.subheader("Final Diagnosis Results")

        st.markdown(
            f'<div class="prediction-box">'
            f'<h4>Predicted Stage: {pred["stage_name"]}</h4>'
            f'<p>Confidence: <b>{pred["confidence"]:.2%}</b></p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        col_chart, col_heatmap = st.columns(2)
        
        with col_chart:
            st.markdown("<h5>Confidence by Stage</h5>", unsafe_allow_html=True)
            df = pd.DataFrame(pred['all_confidences'])
            st.bar_chart(df, x="Stage", y="Confidence")

        with col_heatmap:
            st.markdown("<h5>Explainability: What the AI is looking at</h5>", unsafe_allow_html=True)
            st.image(
                pred['heatmap'],
                caption="A heatmap shows the areas the model focused on.",
                use_container_width=True
            )
            

        st.markdown("---")

        st.subheader("Step 3: Learn More")
        st.markdown("For a better understanding of this stage, explore the links below:")
        links = RELATED_LINKS.get(pred['stage'], [])
        for link in links:
            st.markdown(f"<li><a href='{link['url']}'>{link['text']}</a></li>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)