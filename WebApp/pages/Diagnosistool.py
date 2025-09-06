# pages/1_Diagnosis_Tool.py

import streamlit as st
from PIL import Image
import numpy as np
import time

# --- DUMMY AI FUNCTION ---
# This simulates the AI model's output and is now more detailed.
def get_dummy_prediction(image_array):
    """
    A placeholder function that mimics the model's output.
    It now randomly selects a stage to make the demo more dynamic.
    """
    with st.spinner('AI is analyzing the image...'):
        time.sleep(2) # Simulate a 2-second delay

    # Let's randomly pick a severity for a more dynamic demo
    import random
    stages = [
        {"stage": 0, "stage_name": "No DR", "confidence": 0.98},
        {"stage": 1, "stage_name": "Mild NPDR", "confidence": 0.92},
        {"stage": 2, "stage_name": "Moderate NPDR", "confidence": 0.88},
        {"stage": 3, "stage_name": "Severe NPDR", "confidence": 0.95},
        {"stage": 4, "stage_name": "Proliferative DR", "confidence": 0.91},
    ]
    
    selected_stage = random.choice(stages)
    selected_stage["heatmap"] = Image.open("placeholder_heatmap.png")
    
    return selected_stage

# --- HELPER FUNCTIONS ---
def create_placeholder_image(size=(512, 512), text="Placeholder"):
    img = Image.new('RGB', size, color='grey')
    return img

# Create a placeholder heatmap if it doesn't exist
try:
    placeholder_heatmap = Image.open("placeholder_heatmap.png")
except FileNotFoundError:
    placeholder_heatmap = create_placeholder_image(text="Grad-CAM Heatmap")
    placeholder_heatmap.save("placeholder_heatmap.png")

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
.severity-box.highlight {
    border: 3px solid #000000;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transform: scale(1.05);
}
.severity-0 { background-color: #d4edda; border-color: #c3e6cb; } /* Green */
.severity-1 { background-color: #d4edda; border-color: #c3e6cb; } /* Green */
.severity-2 { background-color: #fff3cd; border-color: #ffeeba; } /* Yellow */
.severity-3 { background-color: #f8d7da; border-color: #f5c6cb; } /* Red */
.severity-4 { background-color: #f8d7da; border-color: #f5c6cb; } /* Red */

.stButton>button {
    height: 3em;
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diagnosis Tool", page_icon="🩺", layout="wide")


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


# --- UI LAYOUT ---
st.title("🩺 Diabetic Retinopathy Diagnosis Interface")
st.markdown(
    "Upload a retinal fundus image to get an AI-powered diagnosis and explanation."
)

# Initialize session state to store the prediction result
if 'prediction' not in st.session_state:
    st.session_state.prediction = None


# --- SIDEBAR ---
with st.sidebar:
    st.title("ICDR Severity Scale")
    st.markdown("The model classifies images according to the following stages:")

    # Display the ICDR scale with colors and highlighting
    for stage, info in ICDR_SCALE_INFO.items():
        highlight_class = "highlight" if st.session_state.prediction and st.session_state.prediction['stage'] == stage else ""
        pointer = "👈" if st.session_state.prediction and st.session_state.prediction['stage'] == stage else ""
        st.markdown(
            f'<div class="severity-box {info["color_class"]} {highlight_class}">'
            f'<b>Stage {stage}:</b> {info["name"]} {pointer}'
            f'</div>',
            unsafe_allow_html=True
        )


# --- MAIN PAGE LAYOUT ---
col1, col2 = st.columns(2)

with col1:
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
        
        st.subheader("🤖 AI Diagnosis Results")
        
        # Determine color based on severity
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
            caption="Grad-CAM Heatmap shows areas the model focused on.",
            use_column_width=True
        )

        # --- DISPLAY RELATED LINKS ---
        st.subheader("📚 Learn More About This Stage")
        links = RELATED_LINKS.get(pred['stage'], [])
        for link in links:
            st.markdown(f"- [{link['text']}]({link['url']})", unsafe_allow_html=True)


with col2:
    st.header("Comparison Image")
    comparison_file = st.file_uploader(
        "Upload a reference image (e.g., a healthy retina)...", type=["jpg", "png", "jpeg"], key="comp_uploader"
    )

    if comparison_file is not None:
        comp_image = Image.open(comparison_file)
        st.image(comp_image, caption="Reference Image", use_column_width=True)
    else:
        st.info("Upload a second image here to compare side-by-side.")