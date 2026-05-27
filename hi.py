import streamlit as st
import joblib

model = joblib.load('model.joblib')

st.title("Hello")

st.header("hey")