# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Web-based artificial intelligence application for predicting sepsis among trauma patients: an international validated cohort study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Abdomineinjury = st.sidebar.selectbox("Abdominal trauma", ("No", "Yes"))
Openinjury = st.sidebar.selectbox("Open trauma", ("No", "Yes"))
Smoking = st.sidebar.selectbox("Smoking", ("No", "Yes"))
ISS = st.sidebar.slider("ISS", 15, 50)
SOFA = st.sidebar.slider("SOFA", 0, 18)
GCS = st.sidebar.slider("GCS", 3, 15)
Redbloodcelcount = st.sidebar.slider("Red blood cell count(×10^12/L)", 1.00, 6.00)
Heartrate = st.sidebar.slider("Heart rate (BPM)", 30, 160)
Respiratoryrate = st.sidebar.slider("Respiratory rate (BPM)", 10, 40)
Hct = st.sidebar.slider("Hematocrit", 0.1000, 0.6000)
Totalprotein = st.sidebar.slider("Total protein (g/L)", 20.0, 80.0)


if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Gender, Abdomineinjury, Openinjury, Smoking, ISS, SOFA, GCS, Redbloodcelcount, Heartrate, Respiratoryrate, Hct, Totalprotein]],
                     columns=["Gender", "Abdomineinjury", "Openinjury", "Smoking", "ISS", "SOFA", "GCS", "Redbloodcelcount", "Heartrate", "Respiratoryrate", "Hct", "Totalprotein"])
    x = x.replace(["Male", "Female"], [1, 0])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Probability of sepsis: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.254:
        st.success(f"Risk group: low-risk group")
    else:
        st.success(f"Risk group: High-risk group")
    if prediction < 0.254:
        st.markdown(f"Recommendation for the low-risk group: Routine infection prevention measures, such as hand hygiene, adequate disinfection of equipment, and isolation of patients with multidrug-resistant organisms, are essential in preventing the spread of infections in the ICU.")
    else:
        st.markdown(f"Recommendation for the high-risk group: Early prevention of sepsis development is crucial to improve patient outcomes. Current prevention strategies for trauma-related infection/sepsis include infection prevention through surgical management, prophylactic antibiotics, tetanus vaccination, and immunomodulatory interventions, as well as organ dysfunction prevention through pharmaceuticals, temporary intravascular shunts, lung-protective strategies, enteral immunonutrition, and acupuncture. In addition, timely administration of antibiotics, fluid resuscitation, and source control can improve outcomes and prevent the development of septic shock.")
st.subheader('Model information')
st.markdown('This AI application, freely accessible and exclusively intended for research purposes, has been specifically designed to comprehensively assess the risk of sepsis in critically injured patients within intensive care units (ICUs). By leveraging machine learning techniques and the scientific literature, this tool may empower healthcare professionals to make more accurate and timely decisions, ultimately improving patient outcomes and reducing mortality rates.')
