import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NITA Academic Analytics - Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Admission Probability Predictor")
st.markdown("### NITA Academic Analytics — MLOps Deployment")
st.write("Adjust the student academic metrics on the left panel to estimate admission probability in real-time.")

st.sidebar.header("User Parameter Inputs")

cgpa = st.sidebar.slider("CGPA (Out of 10.0)", min_value=6.0, max_value=10.0, value=8.5, step=0.01)
gre_score = st.sidebar.slider("GRE Score", min_value=290, max_value=340, value=315, step=1)
toefl_score = st.sidebar.slider("TOEFL Score", min_value=92, max_value=120, value=105, step=1)
uni_rating = st.sidebar.selectbox("University Rating", options=[1, 2, 3, 4, 5], index=2)
sop = st.sidebar.slider("Statement of Purpose (SOP Rating)", min_value=1.0, max_value=5.0, value=3.5, step=0.5)
lor = st.sidebar.slider("Letter of Recommendation (LOR Rating)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
research = st.sidebar.radio("Research Experience", options=["No (0)", "Yes (1)"], index=1)

research_val = 1 if "Yes" in research else 0

predict_chance = (
    -1.2727 +
    (0.0019 * gre_score) +
    (0.0030 * toefl_score) +
    (0.0060 * uni_rating) +
    (0.0016 * sop) +
    (0.0169 * lor) +
    (0.1184 * cgpa) +
    (0.0243 * research_val)
)

predict_chance = float(np.clip(predict_chance, 0.0, 1.0))

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Estimated Chance of Admission")
    st.metric(label="Calculated Probability", value=f"{predict_chance * 100:.1f}%")
    
    if predict_chance >= 0.80:
        st.success("High Probability of Admission! (Tier 1 Match)")
    elif predict_chance >= 0.60:
        st.warning("Moderate Probability of Admission. (Target Match)")
    else:
        st.error("Low Probability of Admission. (Reach University)")

with col2:
    st.subheader("Feature Importance Distribution")
    features = ['CGPA', 'GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'University Rating', 'Research']
    importance = [0.796, 0.088, 0.038, 0.028, 0.024, 0.015, 0.011]
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(features[::-1], importance[::-1], color='#1f77b4')
    ax.set_xlabel('Relative Impact Weight')
    ax.set_title('SageMaker Canvas AutoML Feature Ranking')
    st.pyplot(fig)

st.markdown("---")
st.caption("NITA Academic Analytics © 2026 | Cloud MLOps Architecture Demo")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Graduate Admission Predictor", layout="centered")

st.title("🎓 Graduate Admission Chance Predictor")
st.write("Enter your academic parameters to calculate your estimated admission chance.")

@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Admission_Predict.csv"
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    
    X = df[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
    y = df['Chance of Admit']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()

# User inputs
gre = st.slider("GRE Score", 290, 340, 315)
toefl = st.slider("TOEFL Score", 92, 120, 105)
rating = st.selectbox("University Rating", [1, 2, 3, 4, 5], index=2)
sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, 0.5)
lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.5, 0.5)
cgpa = st.number_input("CGPA (out of 10)", min_value=6.0, max_value=10.0, value=8.5, step=0.1)
research = st.radio("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

if st.button("Predict Admission Chance"):
    features = np.array([[gre, toefl, rating, sop, lor, cgpa, research]])
    prediction = model.predict(features)[0]
    chance_percent = round(prediction * 100, 2)
    
    st.markdown("---")
    st.subheader(f"🎯 Estimated Admission Chance: **{chance_percent}%**")