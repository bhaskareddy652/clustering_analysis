import streamlit as st
import pandas as pd
import numpy as np
st.title('Machine Learning')
st.write('This app builds a machine learning models! ')
url = "https://raw.githubusercontent.com/bhaskareddy652/cluster-deploy/main/Cleaned_dataset.xlsx"
df = pd.read_excel(url)
df
