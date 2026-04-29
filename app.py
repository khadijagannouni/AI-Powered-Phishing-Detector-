
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from src.email_processor import EmailProcessor
from src.feature_extractor import FeatureExtractor
from src.phishing_classifier import PhishingClassifier
from src.explainability_module import ExplainabilityModule
from src.attack_simulator import AttackSimulator
from src.evaluation_report import EvaluationReport

# Page config
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🎣",
    layout="wide",
)

# Load cached artifacts
@st.cache_resource
def load_pipeline():
    classifier = PhishingClassifier()
    classifier.load("models/classifier.pkl")
    extractor: FeatureExtractor = joblib.load("models/feature_extractor.pkl")
    processor = EmailProcessor()
    explainability = ExplainabilityModule()
    return processor, extractor, classifier, explainability


def safe_load():
    try:
        return load_pipeline(), None
    except Exception as e:
        return None, str(e)


pipeline, load_err = safe_load()

# Sidebar navigation
st.sidebar.title("🎣 AI Phishing Detector")
page = st.sidebar.radio(
    "Navigate",
    ["📧 Classify Email", "🔍 Explainability", "⚔️ Attack Simulation", "📊 Analytics Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Kaggle Phishing Email Dataset · scikit-learn · LIME · SHAP")

# PAGE 1: Classify Email

if page == "📧 Classify Email":
    st.title("📧 Email Classification")
    st.markdown("Paste a suspicious email below and click **Analyse** to classify it.")

    col1, col2 = st.columns([3, 1])
    with col1:
        sender  = st.text_input("Sender address", placeholder="noreply@paypa1-secure.com")
        subject = st.text_input("Subject", placeholder="Urgent: Your account has been suspended")
        body    = st.text_area("Email body", height=200,
                               placeholder="Dear customer, click here immediately to verify…")
    with col2:
        st.markdown("####  ")
        analyse_btn = st.button("🔎 Analyse", use_container_width=True, type="primary")
        st.caption("Classification runs locally on your trained model.")

    if analyse_btn:
        if not body.strip():
            st.warning("Please enter an email body.")
        elif load_err:
            st.error(f"Model not loaded – run `python train.py` first.\n\nDetails: {load_err}")
        else:
            processor, extractor, classifier, explainability = pipeline
            raw_text = f"{sender} {subject} {body}"

            with st.spinner("Analysing…"):
                cleaned   = processor.preprocess(raw_text)
                features  = extractor.transform(cleaned, raw_text)
                label     = classifier.predict(features)
                confidence = classifier.confidence

            # Result banner
            color = "#FF4B4B" if label == "Phish" else "#21C354"
            st.markdown(
                f"""<div style="background-color:{color};padding:16px;border-radius:10px;text-align:center">
                    <h2 style="color:white;margin:0">{'🚨 PHISHING DETECTED' if label=='Phish' else '✅ LEGITIMATE EMAIL'}</h2>
                    <p style="color:white;font-size:18px">Confidence: {confidence*100:.1f}%</p>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown("---")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                title={"text": f"Model Confidence – {label}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#FF4B4B" if label == "Phish" else "#21C354"},
                    "steps": [
                        {"range": [0,  50], "color": "#f0f0f0"},
                        {"range": [50, 75], "color": "#ffe0b2"},
                        {"range": [75, 100], "color": "#ffcdd2"},
                    ],
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

            # Store result in session for analytics
            if "session_results" not in st.session_state:
                st.session_state.session_results = []
            st.session_state.session_results.append(
                {"label": label, "confidence": confidence, "snippet": body[:60]}
            )

# PAGE 2: Explainability

elif page == "🔍 Explainability":
    st.title("🔍 LIME & SHAP Explainability")
    st.markdown(
        "LIME highlights **which words** pushed the model towards Phish/Ham.  \n"
        "SHAP shows **global feature importance** across the dataset."
    )

    if load_err:
        st.error(f"Model not loaded. Run `python train.py` first.\nDetails: {load_err}")
        st.stop()

    processor, extractor, classifier, explainability = pipeline

    email_input = st.text_area("Paste email text for LIME explanation", height=150)
    run_lime = st.button("Generate LIME Explanation", type="primary")

    if run_lime and email_input.strip():
        with st.spinner("Running LIME (this takes ~10 seconds)…"):
            # Build a predict_fn compatible with LIME (text → prob array)
            def predict_fn(texts):
                probs = []
                for t in texts:
                    cleaned = processor.preprocess(t)
                    feats   = extractor.transform(cleaned, t)
                    p = classifier.model.predict_proba(feats.reshape(1, -1))[0]
                    probs.append(p)
                return np.array(probs)

            token_weights = explainability.explain_lime(email_input, predict_fn)
            highlighted   = explainability.highlight_tokens(email_input, token_weights)

        st.markdown("#### Token-level LIME Explanation")
        st.markdown(
            f'<div style="line-height:2;font-size:15px">{highlighted}</div>',
            unsafe_allow_html=True,
        )
        st.caption("🔴 Red = phishing signal  |  🟢 Green = ham signal")
        st.markdown("---")

        # Bar chart of top features
        sorted_weights = sorted(token_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        tokens_, weights_ = zip(*sorted_weights)
        colors = ["#FF4B4B" if w > 0 else "#21C354" for w in weights_]
        fig = go.Figure(go.Bar(
            x=list(weights_), y=list(tokens_),
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(title="Top 15 LIME Feature Weights", xaxis_title="Weight", yaxis_title="Token")
        st.plotly_chart(fig, use_container_width=True)

# PAGE 3: Attack Simulation

elif page == "⚔️ Attack Simulation":
    st.title("⚔️ Attack Simulation")
    st.markdown(
        "Generate synthetic phishing emails via LLM and test how well the classifier detects them."
    )

    if load_err:
        st.error(f"Model not loaded. Run `python train.py` first.\nDetails: {load_err}")
        st.stop()

    processor, extractor, classifier, _ = pipeline

    col1, col2 = st.columns(2)
    with col1:
        attack_type  = st.selectbox("Attack type", ["generic_phishing", "spear_phishing", "whaling"])
        impersonate  = st.text_input("Impersonate entity", value="PayPal Security Team")
        n_variants   = st.slider("Number of variants", 1, 5, 3)
    with col2:
        llm_provider = st.selectbox("LLM provider", ["openai", "claude"])
        role         = st.text_input("Target role (spear/whaling)", value="employee")
        company      = st.text_input("Target company", value="Acme Corp")

    api_warning = ""
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        api_warning = "⚠️  OPENAI_API_KEY not set."
    if llm_provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        api_warning = "⚠️  ANTHROPIC_API_KEY not set."
    if api_warning:
        st.warning(api_warning)

    simulate_btn = st.button("🚀 Generate & Classify", type="primary")

    if simulate_btn:
        simulator = AttackSimulator(llm_provider=llm_provider)
        with st.spinner(f"Generating {n_variants} variants with {llm_provider}…"):
            variants = simulator.get_variants(
                attack_type=attack_type,
                n=n_variants,
                impersonate=impersonate,
                role=role,
                company=company,
            )

        st.markdown("---")
        for i, variant in enumerate(variants, 1):
            cleaned   = processor.preprocess(variant)
            features  = extractor.transform(cleaned, variant)
            label     = classifier.predict(features)
            confidence = classifier.confidence

            status = "🚨 Detected as Phish" if label == "Phish" else "✅ Evaded – classified as Ham"
            color  = "#FF4B4B" if label == "Phish" else "#FFA500"

            with st.expander(f"Variant {i} — {status}  ({confidence*100:.1f}% confidence)"):
                st.markdown(f'<div style="border-left:4px solid {color};padding-left:12px">{variant}</div>',
                            unsafe_allow_html=True)
                st.caption(f"Label: **{label}** | Confidence: {confidence*100:.1f}%")

# PAGE 4: Analytics Dashboard

elif page == "📊 Analytics Dashboard":
    st.title("📊 Analytics Dashboard")

    results = st.session_state.get("session_results", [])

    if not results:
        st.info("No emails classified yet. Go to **Classify Email** to analyse some emails first.")
    else:
        df_res = pd.DataFrame(results)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Analysed", len(df_res))
        col2.metric("Phishing Detected", int((df_res["label"] == "Phish").sum()))
        col3.metric("Avg Confidence", f"{df_res['confidence'].mean()*100:.1f}%")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            counts = df_res["label"].value_counts()
            fig_pie = px.pie(values=counts.values, names=counts.index,
                             color_discrete_map={"Phish": "#FF4B4B", "Ham": "#21C354"},
                             title="Phish vs Ham Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_hist = px.histogram(df_res, x="confidence", color="label",
                                    nbins=20,
                                    color_discrete_map={"Phish": "#FF4B4B", "Ham": "#21C354"},
                                    title="Confidence Score Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Session Results")
        st.dataframe(df_res, use_container_width=True)

    #Persisted model evaluation metrics 
    st.markdown("---")
    st.markdown("#### Saved Model Metrics")
    try:
        metrics_df = pd.read_csv("models/metrics.csv")
        st.dataframe(metrics_df, use_container_width=True)
    except FileNotFoundError:
        st.caption("No saved metrics found. Run `python train.py` to generate them.")