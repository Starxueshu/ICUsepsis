# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Prediction of sepsis among major trauma patients admitted to ICU using machine learning techniques: An externally validated cohort study in multicenter")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
Abdomineinjury = st.sidebar.selectbox("Abdominal trauma", ("No", "Yes"))
Openinjury = st.sidebar.selectbox("Open trauma", ("No", "Yes"))
Smoking = st.sidebar.selectbox("Smoking", ("No", "Yes"))
ISS = st.sidebar.slider("ISS", 15, 50)
SOFA = st.sidebar.slider("SOFA", 0, 18)
GCS = st.sidebar.slider("GCS", 3, 15)
Redbloodcelcount = st.sidebar.slider("Red blood cell count(×10^9/L)", 1.00, 6.00)
Heartrate = st.sidebar.slider("Heart rate (BMP)", 30, 160)
Respiratoryrate = st.sidebar.slider("Respiratory rate (BMP)", 10, 40)
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
    st.text(f"Probability of severe sleep disturbance: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.254:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.254:
        st.markdown(f"Recommendation: Routine preoperative evaluation and management with XXX.")
    else:
        st.markdown(f".")
st.subheader('Model information')
st.markdown('The model was developed using the XGBoosting machine algorithm, achieving an area under the curve (AUC) of XXX. The external validation of the model resulted in an AUC of XXX. This online calculator is designed to evaluate the risk of intraoperative massive blood loss, specifically among patients with metastatic spinal tumors undergoing decompressive surgery. It is accessible at no cost and intended solely for research purposes.')