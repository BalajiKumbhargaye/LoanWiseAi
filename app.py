import pickle
import pandas as pd
import streamlit as st

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="LoanWise AI | Loan Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Load saved files --------------------
@st.cache_resource
def load_artifacts():
    with open("loan_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("loan_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("loan_columns.pkl", "rb") as f:
        columns = pickle.load(f)
    return model, scaler, columns

try:
    model, scaler, columns = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find loan_model.pkl, loan_scaler.pkl, or loan_columns.pkl. "
        "Keep all three files in the same folder as app.py."
    )
    st.stop()

# -------------------- Styling --------------------
st.markdown("""
<style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #176b87 100%);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {color: white; margin-bottom: .35rem; font-size: 2.2rem;}
    .hero p {color: #e0f2fe; margin-bottom: 0; font-size: 1rem;}
    .section-title {font-size: 1.25rem; font-weight: 700; margin: .5rem 0 1rem 0;}
    .result-card {
        padding: 1.25rem 1.4rem;
        border-radius: 14px;
        border: 1px solid #d7e2ec;
        background: rgba(128, 145, 160, 0.08);
        margin-top: 1rem;
    }
    div.stButton > button {
        width: 100%;
        min-height: 3rem;
        font-weight: 700;
        border-radius: 10px;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,145,160,.25);
        padding: 1rem;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🏦 LoanWise AI")
    st.caption("Loan Approval Prediction")
    st.divider()

    st.markdown("### About this project")
    st.write(
        "This educational machine-learning application uses a trained "
        "Logistic Regression model to estimate a loan application's outcome."
    )

    st.markdown("### Model information")
    st.metric("Test accuracy", "83.33%")
    st.caption("Accuracy measured on the project's held-out test set.")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Input processing:** Manual encoding + StandardScaler")

    st.divider()
    st.caption(
        "For learning and demonstration only. This prediction is not a "
        "real lending decision or financial advice."
    )

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <h1>Loan Approval Prediction</h1>
    <p>Enter applicant details to estimate the model's prediction.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Input form --------------------
st.markdown('<div class="section-title">👤 Applicant Information</div>', unsafe_allow_html=True)

with st.form("loan_prediction_form"):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Personal details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital status", ["Yes", "No"])
        dependents = st.selectbox("Number of dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self-employed", ["No", "Yes"])

    with right:
        st.markdown("#### Financial details")
        applicant_income = st.number_input(
            "Applicant income (monthly)", min_value=0, value=5000, step=500
        )
        coapplicant_income = st.number_input(
            "Co-applicant income (monthly)", min_value=0.0, value=0.0, step=500.0
        )
        loan_amount = st.number_input(
            "Loan amount (dataset units)", min_value=0.0, value=120.0, step=10.0
        )
        loan_term = st.selectbox(
            "Loan term (months)", [360, 300, 240, 180, 120, 60, 36, 12], index=0
        )
        credit_history = st.selectbox(
            "Credit history", [1.0, 0.0],
            format_func=lambda value: "Good / meets guidelines" if value == 1.0
            else "Poor / does not meet guidelines"
        )
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("🔍 Predict Loan Status", type="primary")

# -------------------- Prediction --------------------
if submitted:
    # These encodings match the preprocessing used in the training notebook.
    applicant_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": int(dependents.replace("+", "")),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area_Rural": int(property_area == "Rural"),
        "Property_Area_Semiurban": int(property_area == "Semiurban"),
        "Property_Area_Urban": int(property_area == "Urban"),
    }

    input_df = pd.DataFrame([applicant_data]).reindex(columns=columns, fill_value=0)

    try:
        input_scaled = scaler.transform(input_df)
        prediction = int(model.predict(input_scaled)[0])
        probabilities = model.predict_proba(input_scaled)[0]
        predicted_probability = float(max(probabilities)) * 100

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        if prediction == 1:
            st.success("### Prediction: Loan Approved")
            st.write("The model predicts the **approved** class for these details.")
        else:
            st.error("### Prediction: Loan Not Approved")
            st.write("The model predicts the **not approved** class for these details.")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Predicted class", "Approved" if prediction == 1 else "Not approved")
        with metric_col2:
            st.metric("Model probability for predicted class", f"{predicted_probability:.1f}%")

        st.caption(
            "The probability is the model's estimated probability, not a guarantee "
            "of the real-world loan outcome. The model may be wrong."
        )
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

else:
    st.info("Fill in the applicant details above, then click **Predict Loan Status**.")

# -------------------- Footer --------------------
st.divider()
st.caption(
    "LoanWise AI · College mini-project · Logistic Regression · "
    "Educational demonstration only"
)
