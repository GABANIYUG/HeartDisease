import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CardioSense | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# LOAD TRAINED MODEL AND FEATURE COLUMNS
# --------------------------------------------------
MODEL_PATH = r"Y:\Machine Learning\ML\LogisticRegrassion_heart.pkl"
COLUMNS_PATH = r"Y:\Machine Learning\ML\columns.pkl"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    return model, list(expected_columns)

try:
    model, expected_columns = load_artifacts()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --background: #09111f;
    --panel: #111d30;
    --border: #25344b;
    --text: #edf4ff;
    --muted: #91a2bb;
    --teal: #42e6c3;
    --blue: #7da9ff;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #15334a 0%, transparent 38%),
        radial-gradient(ellipse at 95% 12%, #102b3b 0%, transparent 32%),
        var(--background);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

p, label, [data-testid="stMarkdownContainer"] {
    font-family: 'DM Sans', sans-serif;
}

.hero {
    padding: 34px 36px;
    border: 1px solid #2a435a;
    border-radius: 24px;
    background: linear-gradient(125deg, #14273b 0%, #102337 55%, #12373e 100%);
    margin-bottom: 25px;
    position: relative;
    overflow: hidden;
}

.eyebrow {
    color: var(--teal);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 13px;
}

.hero-title {
    color: #f4f8ff;
    font-size: clamp(30px, 4vw, 43px);
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    letter-spacing: -1.6px;
    line-height: 1.2;
    margin: 0 0 13px 0;
}

.hero-description {
    color: #a9bbd0;
    font-size: 15px;
    line-height: 1.8;
    max-width: 650px;
}

.hero-chip {
    display: inline-block;
    margin-top: 17px;
    padding: 7px 12px;
    color: #b6ffef;
    border: 1px solid #326b69;
    background: #133a40;
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
}

.section-heading {
    color: #f1f6ff;
    font-family: 'Manrope', sans-serif;
    font-size: 19px;
    font-weight: 700;
    margin: 12px 0 4px 0;
}

.section-caption {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 15px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--border) !important;
    border-radius: 18px !important;
    background: rgba(17, 29, 48, 0.78);
}

div[data-testid="stWidgetLabel"] label {
    color: #cbd8e9 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #0c1929 !important;
    border-color: #31445d !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #edf4ff !important;
}

div[data-baseweb="input"] input {
    color: #edf4ff !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] {
    padding-top: 8px;
}

div[data-testid="stForm"] {
    border: none;
    padding: 0;
}

.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    background: linear-gradient(100deg, #42e6c3, #64d9dc) !important;
    color: #071921 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    min-height: 50px;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    box-shadow: 0 6px 25px #42e6c333;
    transform: translateY(-1px);
}

.result-positive {
    background: linear-gradient(110deg, #34202a, #271c2c);
    border: 1px solid #824454;
    padding: 24px;
    border-radius: 18px;
    margin-top: 20px;
}

.result-negative {
    background: linear-gradient(110deg, #113630, #142c36);
    border: 1px solid #327c70;
    padding: 24px;
    border-radius: 18px;
    margin-top: 20px;
}

.result-label {
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: 700;
    color: #a7b8ca;
    text-transform: uppercase;
}

.result-title {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    margin: 8px 0;
    color: #f1f7ff;
}

.result-description {
    font-size: 14px;
    line-height: 1.7;
    color: #c0cddd;
}

.footer {
    text-align: center;
    color: #70839d;
    font-size: 12px;
    padding: 28px 0 4px 0;
}

hr {
    border-color: #26364d !important;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">CARDIOSENSE &nbsp; / &nbsp; ML HEALTH LAB</div>
    <div class="hero-title">Understand your heart.<br>Explore your health.</div>
    <div class="hero-description">
        Enter the clinical measurements below to generate a prediction
        from your trained machine-learning model.
        A clean, simple interface for your heart disease prediction project.
    </div>
    <div class="hero-chip">✦ &nbsp; Logistic Regression Model</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUT FORM
# --------------------------------------------------
with st.form("heart_prediction_form"):

    # PERSONAL DETAILS
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">01 &nbsp; Personal profile</div>'
            '<div class="section-caption">Basic patient information</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 100, 40)

        with col2:
            sex = st.selectbox(
                "Sex",
                ["M", "F"],
                format_func=lambda x: {
                    "M": "Male",
                    "F": "Female"
                }[x]
            )

    st.markdown("")

    # HEART MEASUREMENTS
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">02 &nbsp; Clinical measurements</div>'
            '<div class="section-caption">Blood pressure and blood markers</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            resting_bp = st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=80,
                max_value=200,
                value=120,
                step=1
            )

            cholesterol = st.number_input(
                "Cholesterol (mg/dl)",
                min_value=100,
                max_value=600,
                value=200,
                step=1
            )

        with col2:
            fasting_bs = st.selectbox(
                "Fasting blood sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            max_hr = st.slider(
                "Maximum heart rate",
                min_value=60,
                max_value=220,
                value=150
            )

    st.markdown("")

    # ECG AND SYMPTOMS
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">03 &nbsp; ECG & activity indicators</div>'
            '<div class="section-caption">Clinical indicators used by the model</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            chest_pain = st.selectbox(
                "Chest pain type",
                ["ATA", "NAP", "TA", "ASY"],
                format_func=lambda x: {
                    "ATA": "Atypical Angina (ATA)",
                    "NAP": "Non-Anginal Pain (NAP)",
                    "TA": "Typical Angina (TA)",
                    "ASY": "Asymptomatic (ASY)"
                }[x]
            )

            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"],
                format_func=lambda x: {
                    "Normal": "Normal",
                    "ST": "ST-T wave abnormality",
                    "LVH": "Left ventricular hypertrophy"
                }[x]
            )

            exercise_angina = st.selectbox(
                "Exercise-induced angina",
                ["Y", "N"],
                format_func=lambda x: "Yes" if x == "Y" else "No"
            )

        with col2:
            oldpeak = st.slider(
                "Oldpeak (ST depression)",
                min_value=0.0,
                max_value=6.0,
                value=1.0,
                step=0.1
            )

            st_slope = st.selectbox(
                "ST slope",
                ["Up", "Flat", "Down"],
                format_func=lambda x: {
                    "Up": "Upsloping",
                    "Flat": "Flat",
                    "Down": "Downsloping"
                }[x]
            )

            st.markdown(
                '<div class="section-caption">'
                'Check that your entries use the same definitions as the '
                'dataset used to train the model.'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown("")
    submitted = st.form_submit_button(
        "✦  Generate Prediction",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# Assumes model was trained on UN-SCALED features.
# --------------------------------------------------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Align the user inputs with the exact training columns.
    input_df = pd.DataFrame([raw_input]).reindex(
        columns=expected_columns,
        fill_value=0
    )

    try:
        prediction = int(model.predict(input_df)[0])

        st.markdown("---")
        st.markdown(
            '<div class="section-heading">Your model result</div>'
            '<div class="section-caption">'
            'Prediction generated from the submitted measurements'
            '</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.markdown("""
            <div class="result-positive">
                <div class="result-label">MODEL OUTPUT · CLASS 1</div>
                <div class="result-title">Heart disease indicated</div>
                <div class="result-description">
                    The trained model classified this input as class 1.
                    This result is an educational model prediction, not a
                    medical diagnosis. Please consult a qualified healthcare
                    professional for health concerns.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-negative">
                <div class="result-label">MODEL OUTPUT · CLASS 0</div>
                <div class="result-title">Heart disease not indicated</div>
                <div class="result-description">
                    The trained model classified this input as class 0.
                    This does not rule out heart disease or replace a medical
                    assessment. Consult a qualified healthcare professional
                    for symptoms or concerns.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Optional probability display
        if hasattr(model, "predict_proba") and 1 in model.classes_:
            probability = model.predict_proba(input_df)[
                0, list(model.classes_).index(1)
            ]

            st.markdown("")
            with st.container(border=True):
                st.markdown("**Model-estimated probability for class 1**")
                st.progress(float(probability))
                st.markdown(f"### {probability:.1%}")
                st.caption(
                    "This is the model's estimated probability, not a "
                    "clinically validated measurement of personal risk."
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "Check that the feature columns and preprocessing match "
            "the model used during training."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<div class="footer">
    CARDIOSENSE · STUDENT ML PROJECT<br>
    Educational use only · Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)