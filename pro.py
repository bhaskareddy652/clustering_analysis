import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Title and description
st.title("Global Clustering: Social & Economic Factors")
st.write(
    "Discover which group your country belongs to by providing key economic and social metrics. The app uses K-Means clustering for classification"
)

# Sidebar input for features
st.sidebar.header("Input Feature Values")

# Collect user inputs
user_input = {
    'Birth Rate': st.sidebar.number_input("Birth Rate:", min_value=0.0, max_value=100.0, value=0.01, step=0.1),
    'CO2 Emission': st.sidebar.number_input("CO2 Emission:", min_value=0.0, max_value=1000.0, value=100.0, step=0.1),
    'Days to Start Business': st.sidebar.number_input("Days to Start Business:", min_value=0.0, max_value=100.0, value=20.0, step=1.0),
    'GDP': st.sidebar.number_input("GDP (in billions):", min_value=0.0, max_value=100000.0, value=500.0, step=10.0),
    'Health Exp % GDP': st.sidebar.number_input("Health Expenditure (% of GDP):", min_value=0.0, max_value=100.0, value=5.0, step=0.1),
    'Health Exp/Capita': st.sidebar.number_input("Health Expenditure per Capita:", min_value=0.0, max_value=100000.0, value=1000.0, step=10.0),
    'Infant Mortality Rate': st.sidebar.number_input("Infant Mortality Rate:", min_value=0.0, max_value=100.0, value=10.0, step=0.1),
    'Internet Usage': st.sidebar.number_input("Internet Usage (%):", min_value=0.0, max_value=100.0, value=70.0, step=0.1),
    'Life Expectancy Female': st.sidebar.number_input("Life Expectancy Female:", min_value=0.0, max_value=100.0, value=75.0, step=0.1),
    'Life Expectancy Male': st.sidebar.number_input("Life Expectancy Male:", min_value=0.0, max_value=100.0, value=70.0, step=0.1),
    'Mobile Phone Usage': st.sidebar.number_input("Mobile Phone Usage (%):", min_value=0.0, max_value=100.0, value=80.0, step=0.1),
    'Population Total': st.sidebar.number_input("Population Total (in millions):", min_value=0.0, max_value=2000.0, value=100.0, step=1.0),
    'Population 0-14': st.sidebar.number_input("Population 0-14 (%):", min_value=0.0, max_value=100.0, value=25.0, step=0.1),
    'Population 15-64': st.sidebar.number_input("Population 15-64 (%):", min_value=0.0, max_value=100.0, value=60.0, step=0.1),
    'Population 65+': st.sidebar.number_input("Population 65+ (%):", min_value=0.0, max_value=100.0, value=15.0, step=0.1),
    'Population Urban': st.sidebar.number_input("Urban Population (%):", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
}

# Convert user input into a DataFrame
user_data = pd.DataFrame([user_input])

# Mock dataset (replace with real dataset later)
mock_data = pd.DataFrame(
    np.random.rand(100, len(user_input)) * 100,
    columns=user_input.keys()
)

# Combine user data with mock data
combined_data = pd.concat([mock_data, user_data], ignore_index=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)

# Apply KMeans clustering
n_clusters = 3  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(scaled_data)

# Get cluster label for user input
user_cluster = kmeans.predict(scaled_data[-1].reshape(1, -1))[0]

# Display the result
st.success(f"Your country belongs to Cluster {user_cluster}!")

# Visualize user input
st.write("### Your Feature Values")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(user_input.keys(), user_data.iloc[0].values, color='skyblue')
ax.set_title("Your Country's Feature Values")
ax.set_ylabel("Value")
ax.set_xlabel("Features")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# Cluster visualization
labels = kmeans.labels_
# Cluster descriptions
st.write("### Cluster Descriptions:")
st.write("""
- **Cluster 0**: Developed Economies - High GDP, excellent healthcare, and high life expectancy.
- **Cluster 1**: High-Income Economies - Extremely high GDP, advanced technology, and strong healthcare systems.
- **Cluster 2**: Low-Income Economies - Lower GDP, limited access to healthcare, and shorter life expectancy.
""")
