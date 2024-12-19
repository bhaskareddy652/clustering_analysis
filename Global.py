import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
st.title("Clustering Analysis Web App")
st.write("This app is for machine learing algorithms!!!!")
data_source = st.radio("Choose how to provide the dataset:", 
                       options=["Upload File", "Enter Local File Path"])
if data_source == "Upload File":
    uploaded_file = st.file_uploader("Upload your dataset (Excel file):", type=["xlsx"])
    if uploaded_file:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
elif data_source == "Enter Local File Path":
    # File path input option
    file_path = st.text_input("Enter the file path to your dataset (e.g., C:/path/to/your/file.xlsx):")
    if file_path:
        try:
            df = pd.read_excel(file_path)
            st.write("Dataset Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the file: {e}")
if 'df' in locals():
    # Handle categorical columns
    cat = df.select_dtypes(include='object')
    for col in cat.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    columns = st.multiselect("Select columns for clustering:", df.columns, default=df.columns[:2])
    if not columns:
        st.warning("Please select at least one column for clustering.")
    else:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[columns])
        clustering_method = st.selectbox("Select a Clustering Method", 
                                         ["KMeans", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"])

        if clustering_method == "KMeans":
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters, random_state=101, n_init=10)
            predicted = model.fit_predict(df_scaled)
            score = silhouette_score(df_scaled, predicted)
            st.write(f"Silhouette Score: {score:.2f}")
            inertias = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=101, n_init=10)
                kmeans.fit(df_scaled)
                inertias.append(kmeans.inertia_)
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), inertias, marker='o')
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Inertia')
            st.pyplot(fig)
            fig, ax = plt.subplots()
            scatter = ax.scatter(df_scaled[:, 0], df_scaled[:, 1], c=predicted, cmap='viridis')
            ax.set_title('KMeans Clustering Results')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            plt.colorbar(scatter, ax=ax)
            st.pyplot(fig)
        elif clustering_method == "Agglomerative Clustering":
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            linkage_method = st.selectbox("Select Linkage Method", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            predicted = model.fit_predict(df_scaled)
            score = silhouette_score(df_scaled, predicted)
            st.write(f"Silhouette Score: {score:.2f}")
        elif clustering_method == "DBSCAN":
            eps = st.slider("Select EPS (Distance Threshold)", 0.1, 5.0, 0.5)
            min_samples = st.slider("Select Minimum Samples", 2, 10, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            predicted = model.fit_predict(df_scaled)
            if len(set(predicted)) > 1:
                score = silhouette_score(df_scaled, predicted)
                st.write(f"Silhouette Score: {score:.2f}")
            else:
                st.write("DBSCAN did not find any clusters.")
        elif clustering_method == "Gaussian Mixture":
            n_components = st.slider("Select Number of Components", 2, 10, 3)
            model = GaussianMixture(n_components=n_components, random_state=101)
            predicted = model.fit_predict(df_scaled)
            score = silhouette_score(df_scaled, predicted)
            st.write(f"Silhouette Score: {score:.2f}")
            logs = []
            for i in range(1, 11):
                gm = GaussianMixture(n_components=i, random_state=101)
                gm.fit(df_scaled)
                logs.append(gm.score(df_scaled))
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), logs, marker='o')
            ax.set_title('Log-Likelihood for Gaussian Mixture')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Log-Likelihood')
            st.pyplot(fig)
else:
    st.info("Please provide a dataset to rock!!!")
