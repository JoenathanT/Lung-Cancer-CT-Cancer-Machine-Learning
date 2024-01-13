import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman:', ('eda', 'Predict customer default payment'))

if navigation == 'eda':
    eda.run()
else:
    prediction.run()