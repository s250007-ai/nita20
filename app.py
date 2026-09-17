import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="University Admission Chance Predictor", layout="wide")

# Sidebar inputs
st.sidebar.title("Applicant Profile")
gre = st.sidebar.slider("GRE Score", 290, 340, 316)
toefl = st.sidebar.slider("TOEFL Score", 92, 120, 107)
rating = st.sidebar.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.sidebar.slider("SOP Strength", 1.0, 5.0, 3.5, 0.5)
lor = st.sidebar.slider("LOR Strength", 1.0, 5.0, 3.5, 0.5)
cgpa = st.sidebar.slider("CGPA", 6.8, 9.92, 8.50, 0.01)
research_input = st.sidebar.radio("Research Experience", ["No", "Yes"], index=1)
research = 1 if research_input == "Yes" else 0

# Main Title
st.title("🎓 University Admission Chance Predictor")
st.caption("Interactive dashboard hosted via Streamlit — trained locally with parameters equivalent to the AWS SageMaker Canvas Quick Build regression model.")

# Metrics
col1, col2 = st.columns(2)
col1.metric("Model RMSE", "0.0650")
col2.metric("Model R²", "0.7932")

st.subheader("Predicted Chance of Admit")

# Formula Calculation
pred_chance = (-1.25 + (gre * 0.0018) + (toefl * 0.0028) + (rating * 0.006) + 
               (sop * 0.002) + (lor * 0.017) + (cgpa * 0.118) + (research * 0.024))
pred_chance = float(np.clip(pred_chance, 0.0, 1.0))

# Progress bar and percentage text
st.progress(pred_chance)
st.markdown(f"### **{pred_chance * 100:.2f}%**")

# Charts section
c1, c2 = st.columns(2)

with c1:
    st.markdown("##### **Feature Importance (Column Impact)**")
    features = ['CGPA', 'GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'Research', 'University Rating']
    importance = [0.71, 0.16, 0.03, 0.03, 0.02, 0.02, 0.01]
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(features[::-1], importance[::-1], color='#4672b8')
    ax.set_xlabel("Relative importance", fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    st.pyplot(fig)

with c2:
    st.markdown("##### **Your Profile vs. Dataset Average**")
    categories = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
    dataset_avg = [316.7, 107.2, 3.11, 3.37, 3.48, 8.58, 0.56]
    user_profile = [gre, toefl, rating, sop, lor, cgpa, research]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width/2, dataset_avg, width, label='Dataset avg', color='#1f77b4')
    ax.bar(x + width/2, user_profile, width, label='Your profile', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    st.pyplot(fig)

st.markdown("---")
st.caption("Deployment architecture: PyCharm/Streamlit (Week 14) ➔ GitHub 'nita20' (Week 15) ➔ AWS EC2 + Apache2 reverse-proxy on port 80, and Streamlit Cloud hybrid deploy (Week 16).")
