import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# Title for the app
st.title("Clustering Analysis Web App")

# Input the file path manually
file_path = st.text_input("Enter the file path to your dataset (e.g., C:/path/to/your/file.xlsx)")

if file_path:
    try:
        # Read the uploaded file using the provided file path
        df = pd.read_excel(file_path)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Handle categorical columns
        cat = df.select_dtypes(include='object')
        for col in cat.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Columns to be used for clustering
        columns = ['Birth Rate','CO2 Emissions',
                   'Days to Start Business', 'Ease of Business', 'Energy Usage',
                   'Health Exp % GDP','Hours to do Tax',
                   'Infant Mortality Rate', 'Internet Usage', 'Lending Interest',
                   'Life Expectancy Female', 'Life Expectancy Male', 'Mobile Phone Usage',
                   'Number of Records', 'Population 0-14', 'Population 15-64',
                   'Population 65+', 'Population Total', 'Population Urban',]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df[columns])

        # Clustering Method Selection
        clustering_method = st.selectbox("Select a Clustering Method", 
                                         ["KMeans", "Agglomerative Clustering", "DBSCAN", "Gaussian Mixture"])

        if clustering_method == "KMeans":
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            model1 = KMeans(n_clusters=n_clusters, random_state=101, n_init=1)
            predicted1 = model1.fit_predict(df_scaled)
            score1 = silhouette_score(df_scaled, predicted1)
            st.write(f"Silhouette Score: {score1:.2f}")
            
            # Elbow method plot
            inertias = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, random_state=101, n_init=3)
                kmeans.fit(df_scaled)
                inertias.append(kmeans.inertia_)
            fig1, ax1 = plt.subplots()
            ax1.plot(range(1, 11), inertias, marker='o')
            ax1.set_title('Elbow Method')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Inertia')
            st.pyplot(fig1)

        elif clustering_method == "Agglomerative Clustering":
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            linkage_method = st.selectbox("Select Linkage Method", ["ward", "complete", "average", "single"])
            model2 = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            predicted2 = model2.fit_predict(df_scaled)
            score2 = silhouette_score(df_scaled, predicted2)
            st.write(f"Silhouette Score: {score2:.2f}")

        elif clustering_method == "DBSCAN":
            eps = st.slider("Select EPS (Distance Threshold)", 0.1, 5.0, 0.5)
            min_samples = st.slider("Select Minimum Samples", 2, 10, 5)
            model3 = DBSCAN(eps=eps, min_samples=min_samples)
            predicted3 = model3.fit_predict(df_scaled)
            if len(set(predicted3)) > 1:
                score3 = silhouette_score(df_scaled, predicted3)
                st.write(f"Silhouette Score: {score3:.2f}")
            else:
                st.write("DBSCAN did not find any clusters.")

        elif clustering_method == "Gaussian Mixture":
            n_components = st.slider("Select Number of Components", 2, 10, 3)
            model4 = GaussianMixture(n_components=n_components, random_state=101)
            predicted4 = model4.fit_predict(df_scaled)
            score4 = silhouette_score(df_scaled, predicted4)
            st.write(f"Silhouette Score: {score4:.2f}")
            
            # Elbow method plot for Gaussian Mixture
            logs = []
            for i in range(1, 11):
                gm = GaussianMixture(n_components=i, random_state=101)
                gm.fit(df_scaled)
                logs.append(gm.score(df_scaled))
            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, 11), logs, marker='o')
            ax2.set_title('Elbow Method for Gaussian Mixture')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Log Likelihood')
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error reading the file: {e}")

else:
    st.info("Please enter the file path to your dataset.")
