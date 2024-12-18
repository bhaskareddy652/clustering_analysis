import streamlit as st
import pandas as pd
import numpy as np
st.title('Machine Learning')
st.write('This app builds a machine learning models! ')
df=pd.read_excel("Cleaned_dataset.xlsx")
df
