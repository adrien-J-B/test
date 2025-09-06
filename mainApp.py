# app.py
# This is your main homepage.

import streamlit as st
from PIL import Image

# Set the page configuration
# This must be the first Streamlit command in your script
st.set_page_config(
    page_title="DR-AI-Vision | Home",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Page Content ---

# Header Section
st.title("👁️ AI-Powered Diabetic Retinopathy Detection")
st.markdown("### Helping prevent blindness through early, accessible, and intelligent screening.")

st.markdown("---")

# Main Body
col1, col2 = st.columns(2)

with col1:
    st.header("The Problem: A Silent Threat")
    st.write(
        """
        Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults.
        Early detection is crucial to prevent severe vision loss, but regular screening can be
        inaccessible or expensive for many. Our project aims to bridge that gap.
        """
    )

    st.header("Our Solution: AI for Everyone")
    st.write(
        """
        We've developed an AI tool that analyzes retinal fundus images to detect and grade
        the severity of Diabetic Retinopathy based on the International Clinical Diabetic
        Retinopathy (ICDR) scale. Our goal is to provide a fast, reliable, and explainable
        diagnostic aid for healthcare professionals and patients.
        """
    )
    st.info(
        "**Navigate to the 'Diagnosis Tool' page from the sidebar to try it out!**",
        icon="👈"
    )

with col2:
    # You can replace this with a more relevant image for your project
    st.image(
        "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Retina_display_scan.max-1000x1000.jpg",
        caption="AI analyzing a retinal fundus image."
    )

st.markdown("---")

# Disclaimer Section
st.header("⚠️ Important Disclaimer")
st.warning(
    """
    This tool is a proof-of-concept developed for a hackathon. **It is not a medical device.**
    The predictions are for informational purposes only and should not be used for self-diagnosis
    or to replace professional medical advice. Always consult a qualified healthcare provider
    for any health concerns.
    """
)

# Team Credits in Sidebar
st.sidebar.title("Meet the Team")
st.sidebar.info(
    """
    **Project:** DR-AI-Vision (Hackathon)

    - **UI/UX Person A**
    - **UI/UX Person B**
    - **AI Core Person A**
    - **AI Core Person B**
    - **AI Support/Deployment Person**
    """
)