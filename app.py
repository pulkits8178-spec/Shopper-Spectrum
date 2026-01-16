import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation System")

# -----------------------------
# Load and Clean Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv", encoding="latin1")

    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

df = load_data()

# -----------------------------
# RFM Feature Engineering
# -----------------------------
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
reference_date = df["InvoiceDate"].max()

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (reference_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# -----------------------------
# Scaling & Clustering
# -----------------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

cluster_labels = {
    0: "High-Value",
    1: "Regular",
    2: "Occasional",
    3: "At-Risk"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_labels)

# -----------------------------
# Recommendation System
# -----------------------------
pivot = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    fill_value=0
)

similarity = cosine_similarity(pivot.T)
similarity_df = pd.DataFrame(
    similarity,
    index=pivot.columns,
    columns=pivot.columns
)

def recommend_products(product_name, n=5):
    if product_name not in similarity_df.index:
        return []
    scores = similarity_df[product_name].sort_values(ascending=False)[1:n+1]
    return scores.index.tolist()

# -----------------------------
# Streamlit UI
# -----------------------------
tab1, tab2 = st.tabs(["🛍 Product Recommendation", "👥 Customer Segmentation"])

# -------- TAB 1 ---------------
with tab1:
    st.header("🛍 Product Recommendation")

    product_name = st.text_input(
        "Product Name डालो",
        "WHITE HANGING HEART T-LIGHT HOLDER"
    )

    if st.button("Get Recommendations"):
        recs = recommend_products(product_name)

        if recs:
            st.success("Top 5 Similar Products:")
            for i, p in enumerate(recs, 1):
                st.write(f"{i}. {p}")
        else:
            st.error("Product database में नहीं मिला")

# -------- TAB 2 ---------------
with tab2:
    st.header("👥 Customer Segmentation (RFM)")

    recency = st.number_input("Recency (days)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Customer Segment"):
        user_data = pd.DataFrame(
            [[recency, frequency, monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )

        user_scaled = scaler.transform(user_data)
        cluster = kmeans.predict(user_scaled)[0]
        segment = cluster_labels[cluster]

        st.success(f"Customer Segment: **{segment}**")

