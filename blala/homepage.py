import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle

st.set_page_config(page_title="Campus Mental Health Assistant", page_icon="🏠")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "Risk Predictor", "Resources Q&A"])

if page == "Home":
    st.title("Home Page")
    st.header("Welcome to the Home Page! 🏠")
    st.write("This app helps predict student burnout risk and provides mental health resources.")
    
    # Add some dashboard elements
    st.subheader("Dashboard Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", "150", "25% from last month")
    
    with col2:
        st.metric("At Risk", "42", "-8% from last month")
    
    with col3:
        st.metric("Intervention Success", "78%", "12% improvement")
    
    # Sample chart
    st.subheader("Risk Distribution")
    chart_data = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Students': [108, 32, 10]
    })
    st.bar_chart(chart_data.set_index('Risk Level'))
    
    # Quick actions
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("📊 View Reports"):
            st.switch_page("pages/risk_predictor.py")
    
    with action_col2:
        if st.button("❓ Ask Questions"):
            st.switch_page("pages/qa_retrieval.py")
    
    with action_col3:
        if st.button("📋 Upload Data"):
            st.switch_page("pages/risk_predictor.py")

elif page == "Risk Predictor":
    # Import your risk predictor page
    from risk_predictor import main as risk_predictor_main
    risk_predictor_main()

elif page == "Resources Q&A":
    # Import your Q&A page
    from qa_retrieval import main as qa_main
    qa_main()