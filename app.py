import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="AutoDataGuard | ML Anomaly Engine", layout="wide")

st.title("🛡️ AutoDataGuard: Automated ML Data Quality & Anomaly Detection")
st.markdown(
    "**Enterprise AI Data Operations Tool** — Automatically audits datasets, detects missing/corrupted records, "
    "and isolates statistical anomalies using Unsupervised Machine Learning (**Isolation Forest**)."
)

# Sidebar Controls
st.sidebar.header("⚙️ Model & Data Controls")
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Use Built-in Enterprise Sample Data", "Upload Custom CSV"]
)
contamination = st.sidebar.slider(
    "Expected Anomaly Rate (Contamination)", 
    min_value=0.01, max_value=0.20, value=0.05, step=0.01
)

@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_normal = 475
    n_anomalies = 25
    
    # Normal e-commerce / financial transaction operations data
    normal_amount = np.random.normal(loc=2500, scale=600, size=n_normal).clip(100, 6000)
    normal_latency = np.random.normal(loc=120, scale=25, size=n_normal).clip(30, 250)
    normal_items = np.random.poisson(lam=4, size=n_normal).clip(1, 15)
    
    # Anomalous spikes / corrupted records
    anom_amount = np.random.uniform(low=12000, high=28000, size=n_anomalies)
    anom_latency = np.random.uniform(low=600, high=1500, size=n_anomalies)
    anom_items = np.random.randint(low=25, high=60, size=n_anomalies)
    
    df = pd.DataFrame({
        "Transaction_ID": [f"TXN-{1000+i}" for i in range(n_normal + n_anomalies)],
        "Order_Value_INR": np.concatenate([normal_amount, anom_amount]).round(2),
        "API_Latency_ms": np.concatenate([normal_latency, anom_latency]).round(1),
        "Items_Count": np.concatenate([normal_items, anom_items]),
        "Region": np.random.choice(["Chennai", "Bengaluru", "Mumbai", "Hyderabad"], size=n_normal + n_anomalies)
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# Load Data
if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("👆 Please upload a CSV file in the sidebar, or switch to the Built-in Enterprise Sample Data.")
        st.stop()
else:
    df = generate_sample_data()

# Select Numeric Features for ML
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Dataset must contain at least 2 numeric columns for ML Anomaly Detection.")
    st.stop()

st.sidebar.subheader("📊 Feature Selection")
selected_features = st.sidebar.multiselect(
    "Features for Isolation Forest:", numeric_cols, default=numeric_cols[:3]
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 numeric features in the sidebar.")
    st.stop()

# Run Isolation Forest ML Model
clean_df = df.copy()
clean_df[selected_features] = clean_df[selected_features].fillna(clean_df[selected_features].median())

model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
preds = model.fit_predict(clean_df[selected_features])
scores = model.decision_function(clean_df[selected_features])

clean_df["ML_Audit_Status"] = np.where(preds == -1, "🔴 Anomaly Detected", "🟢 Normal Record")
clean_df["Anomaly_Score"] = scores.round(4)

# Top KPI Cards
total_rows = len(clean_df)
anomaly_count = int((preds == -1).sum())
normal_count = total_rows - anomaly_count
missing_values = int(df.isna().sum().sum())
health_score = round(((total_rows - anomaly_count) / total_rows) * 100, 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records Audited", f"{total_rows:,}")
col2.metric("ML Anomalies Flagged", f"{anomaly_count}", delta="-Action Required", delta_color="inverse")
col3.metric("Missing Values Imputed", f"{missing_values}")
col4.metric("Dataset Quality Score", f"{health_score}%")

st.divider()

# Interactive Visualization & Data Tabs
tab1, tab2, tab3 = st.tabs(["📈 Interactive ML Scatter Plot", "🚨 Flagged Anomalies Table", "✅ Clean Export Ready Data"])

with tab1:
    st.subheader(f"Unsupervised Isolation Forest Cluster: {selected_features[0]} vs {selected_features[1]}")
    fig = px.scatter(
        clean_df,
        x=selected_features[0],
        y=selected_features[1],
        color="ML_Audit_Status",
        color_discrete_map={"🟢 Normal Record": "#00CC96", "🔴 Anomaly Detected": "#EF553B"},
        hover_data=clean_df.columns,
        title="Red Points Indicate Statistically Corrupted or Fraudulent Outliers"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🚨 Records Flagged by ML for Human-in-the-Loop Review")
    anomalies_only = clean_df[clean_df["ML_Audit_Status"] == "🔴 Anomaly Detected"]
    st.dataframe(anomalies_only, use_container_width=True)

with tab3:
    st.subheader("✅ Sanitized Dataset (Anomalies Removed)")
    verified_df = clean_df[clean_df["ML_Audit_Status"] == "🟢 Normal Record"].drop(columns=["ML_Audit_Status", "Anomaly_Score"])
    st.dataframe(verified_df.head(50), use_container_width=True)
    csv_export = verified_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Verified Clean CSV",
        data=csv_export,
        file_name="verified_clean_dataset.csv",
        mime="text/csv"
    )
  
