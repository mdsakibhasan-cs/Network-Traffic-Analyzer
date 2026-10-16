import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.preprocessing import split_features_and_labels
from src.feature_engineering import add_derived_features
from src.classification import train_traffic_classifier, predict_traffic
from src.anomaly_detection import train_anomaly_detector, detect_anomalies
from src.risk_scoring import calculate_risk_score


st.set_page_config(
    page_title="Network Traffic Analyzer",
    page_icon="🛡️",
    layout="wide",
)


st.title("🛡️ Network Traffic Analyzer")
st.caption(
    "Network traffic classification, anomaly detection, and risk analysis"
)


@st.cache_data
def load_dataset():
    file_path = "data/cic_ids2017_sample.csv"

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df["Label"] = df["Label"].astype(str).str.strip()

    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def run_ml_pipeline():

    df = load_dataset()

    features, labels = split_features_and_labels(df)

    features = add_derived_features(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    classifier = train_traffic_classifier(
        X_train,
        y_train,
    )

    predicted_labels = predict_traffic(
        classifier,
        X_test,
    )

    classification_accuracy = accuracy_score(
        y_test,
        predicted_labels,
    )

    anomaly_detector = train_anomaly_detector(
        X_train,
        contamination=0.05,
    )

    anomaly_results = detect_anomalies(
        anomaly_detector,
        X_test,
    )

    risk_results = calculate_risk_score(
        anomaly_results["anomaly_score"],
        predicted_labels,
    )

    results = pd.DataFrame(
        {
            "actual_label": y_test.values,
            "predicted_label": predicted_labels.values,
            "status": anomaly_results["status"].values,
            "anomaly_score": anomaly_results[
                "anomaly_score"
            ].values,
            "risk_score": risk_results[
                "risk_score"
            ].values,
            "risk_level": risk_results[
                "risk_level"
            ].values,
        }
    )

    return (
        df,
        results,
        classification_accuracy,
    )


df, results, classification_accuracy = run_ml_pipeline()


st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Traffic Analysis",
        "Anomaly Detection",
        "Risk Analysis",
        "Suspicious Flows",
    ],
)


if page == "Overview":

    st.header("System Overview")

    total_flows = len(df)
    benign_count = (df["Label"] == "BENIGN").sum()
    ddos_count = (df["Label"] == "DDoS").sum()

    anomaly_count = (
        results["status"] == "Anomaly"
    ).sum()

    high_risk_count = (
        results["risk_level"] == "High"
    ).sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Flows",
        f"{total_flows:,}",
    )

    col2.metric(
        "Benign Traffic",
        f"{benign_count:,}",
    )

    col3.metric(
        "DDoS Traffic",
        f"{ddos_count:,}",
    )

    col4.metric(
        "Anomalies",
        f"{anomaly_count:,}",
    )

    col5.metric(
        "High Risk",
        f"{high_risk_count:,}",
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Traffic Distribution")

        label_counts = (
            df["Label"]
            .value_counts()
            .reset_index()
        )

        label_counts.columns = [
            "Label",
            "Count",
        ]

        fig = px.bar(
            label_counts,
            x="Label",
            y="Count",
            title="Traffic Classes",
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    with col2:

        st.subheader("Risk Distribution")

        risk_counts = (
            results["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Count",
        ]

        fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            title="Risk Levels",
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    st.subheader("Model Performance")

    st.metric(
        "Random Forest Test Accuracy",
        f"{classification_accuracy:.2%}",
    )

    st.caption(
        "Evaluation uses a balanced 2,000-flow sample "
        "from the CIC-IDS2017 DDoS dataset."
    )


elif page == "Traffic Analysis":

    st.header("📊 Traffic Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    selected_feature = st.selectbox(
        "Select traffic feature",
        numeric_columns,
    )

    fig = px.histogram(
        df,
        x=selected_feature,
        color="Label",
        title=f"{selected_feature} Distribution",
    )

    st.plotly_chart(
        fig,
        width='stretch',
    )

    if "Protocol" in df.columns:

        st.subheader("Protocol Distribution")

        protocol_counts = (
            df["Protocol"]
            .value_counts()
            .reset_index()
        )

        protocol_counts.columns = [
            "Protocol",
            "Count",
        ]

        fig = px.bar(
            protocol_counts,
            x="Protocol",
            y="Count",
            title="Network Protocol Distribution",
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )


elif page == "Anomaly Detection":

    st.header("🚨 Anomaly Detection")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Normal Flows",
            int(
                (
                    results["status"] == "Normal"
                ).sum()
            ),
        )

    with col2:

        st.metric(
            "Anomalous Flows",
            int(
                (
                    results["status"] == "Anomaly"
                ).sum()
            ),
        )

    st.subheader("Anomaly Score Distribution")

    fig = px.histogram(
        results,
        x="anomaly_score",
        color="status",
        title="Isolation Forest Anomaly Scores",
    )

    st.plotly_chart(
        fig,
        width='stretch',
    )

    st.subheader("Detected Anomalies")

    anomalies = results[
        results["status"] == "Anomaly"
    ].sort_values(
        "anomaly_score",
        ascending=False,
    )

    st.dataframe(
        anomalies,
        width='stretch',
        hide_index=True,
    )


elif page == "Risk Analysis":

    st.header("⚠️ Risk Analysis")

    risk_counts = (
        results["risk_level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0,
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count",
    ]

    fig = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        title="Risk Level Distribution",
    )

    st.plotly_chart(
        fig,
        width='stretch',
    )

    st.subheader("High-Risk Traffic")

    high_risk = results[
        results["risk_level"] == "High"
    ].sort_values(
        "risk_score",
        ascending=False,
    )

    st.dataframe(
        high_risk,
        width='stretch',
        hide_index=True,
    )


elif page == "Suspicious Flows":

    st.header("🔎 Suspicious Traffic")

    suspicious = results[
        (
            results["predicted_label"]
            != "BENIGN"
        )
        |
        (
            results["status"]
            == "Anomaly"
        )
    ].sort_values(
        "risk_score",
        ascending=False,
    )

    st.metric(
        "Suspicious Flows",
        f"{len(suspicious):,}",
    )

    st.dataframe(
        suspicious,
        width='stretch',
        hide_index=True,
    )

    st.subheader("Top 10 Highest-Risk Flows")

    st.dataframe(
        suspicious.head(10),
        width='stretch',
        hide_index=True,
    )




