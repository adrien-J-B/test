# app.py
# This is your main homepage.

import streamlit as st
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="GlucoVision | Home",
    page_icon="favicon1.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
        padding: 0 4rem; /* Adjusted padding for main content */
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("https://images.unsplash.com/photo-1601614051016-56a89c35a8f4") no-repeat center center/cover; /* Replaced with abstract medical image */
        height: 60vh;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-align: center;
        padding: 0 2rem;
    }
    .hero-section h1 {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    .hero-section h3 {
        font-size: 1.5rem;
        font-weight: 400;
        color: white;
        margin-top: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }
    
    /* Section Titles */
    .section-title h2 {
        font-size: 2.5rem;
        font-weight: 500;
        color: #004d99;
        text-align: center;
        margin: 4rem 0 2rem;
    }
    
    /* Content Blocks */
    .content-block {
        background-color: #ffffff;
        padding: 2.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        height: 100%; /* Ensure consistent height */
    }
    .content-block h4 {
        color: #004d99;
        font-weight: 500;
        margin-top: 0;
    }
    .content-block p {
        color: #666666;
        line-height: 1.6;
    }
    
    /* Image Styling */
    .st-emotion-cache-183-b7-e2 img {
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); /* Subtle shadow for images */
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
    
    /* Disclaimer Section */
    .disclaimer-section {
        background-color: #e0e6ed; /* Lighter gray for disclaimer */
        padding: 2rem;
        border-radius: 8px;
        margin: 4rem 0;
    }
    .disclaimer-section h5 {
        color: #d9534f;
        font-weight: 600;
    }
    
    /* Sidebar styling for team credits */
    [data-testid="stSidebar"] {
        background-color: #004d99;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: white;
    }
    [data-testid="stSidebar"] .st-emotion-cache-183-b7-e2 {
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Page Content ---
# Hero Section
st.markdown("""
<div class="hero-section">
    <div>
        <h1>AI-Powered Diabetic Retinopathy Detection</h1>
        <h3>Helping prevent blindness through early, accessible, and intelligent screening.</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
st.markdown("<div class='section-title'><h2>Our Mission</h2></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.image("https://images.unsplash.com/photo-1599092305886-07971b86d997?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w0NTUyMzh8MHwxfHNlYXJjaHwxNXx8cmV0aW5hJTIwZG9jdG9yfGVufDB8fHx8MTcwMzEyMzE1OHww&ixlib=rb-4.0.3&q=80&w=1080", caption="A medical professional using advanced technology for diagnosis.", use_column_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='content-block'>", unsafe_allow_html=True)
    st.markdown("<h4>The Problem: A Silent Threat</h4>", unsafe_allow_html=True)
    st.write(
        """
        Diabetic Retinopathy (DR) is a leading cause of blindness among working-age adults. 
        Early detection is crucial to prevent severe vision loss, but regular screening can be
        inaccessible or expensive for many. Our project aims to bridge that gap.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='content-block'>", unsafe_allow_html=True)
    st.markdown("<h4>Our Solution: AI for Everyone</h4>", unsafe_allow_html=True)
    st.write(
        """
        We've developed an AI tool that analyzes retinal fundus images to detect and grade
        the severity of Diabetic Retinopathy based on the International Clinical Diabetic
        Retinopathy (ICDR) scale. Our goal is to provide a fast, reliable, and explainable
        diagnostic aid for healthcare professionals and patients.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.page_link("pages/Diagnosistool.py", label="Get Started with Diagnosis", icon="🩺")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Disclaimer Section
st.markdown("<div class='disclaimer-section'>", unsafe_allow_html=True)
st.markdown("<h5>⚠️ Important Disclaimer</h5>", unsafe_allow_html=True)
st.warning(
    """
    This tool is a proof-of-concept developed for a hackathon. **It is not a medical device.**
    The predictions are for informational purposes only and should not be used for self-diagnosis
    or to replace professional medical advice. Always consult a qualified healthcare provider
    for any health concerns.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

# Team Credits in Sidebar
with st.sidebar:
    st.markdown("<h2>Meet the Team</h2>", unsafe_allow_html=True)
    st.info(
        """
        **** 

        - **Arya**
        - **Yogendra**
        - **Yuvraj**
        - **Adrien**
        - **Casseeram**
        """
    )