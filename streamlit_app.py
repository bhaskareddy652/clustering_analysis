import streamlit as st
import pandas as pd
import numpy as np
st.title('Machine Learning')
st.write('This app builds a machine learning models! ')
df=pd.read_excel("https://raw.githubusercontent.com/bhaskareddy652/cluster-deploy/bbf9c0acc66dcbbfe985578266543e165ebe349b/Cleaned_dataset.xlsx")
df
