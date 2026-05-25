import streamlit as st
import pandas as pd
import joblib
import json

# Load the model and categories once when the app starts
model = joblib.load('model.joblib')

with open('categories.json', 'r') as f:
    categories = json.load(f)

st.title("Job Salary Predictor")

# Input form
with st.form("salary_form"):
    City = st.selectbox("City", options=categories['cities'], index=None)
    Company = st.selectbox("Company", options=categories['companies'], index=None)
    Job_Title = st.selectbox("Job Title", options=categories['job titles'], index=None)
    Industry = st.selectbox("Industry", options=categories['industries'], index=None)
    Job_Type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract"], index=None)
    Education_Required = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"], index=None)
    Location_Tier = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"], index=None)
    Work_Mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "Office"], index=None)
    Company_Type = st.selectbox("Company Type", ["Startup", "MNC", "SME"], index=None)
    Experience_Level = st.selectbox("Experience Level", 
                                    ['Fresher (0-1 yr)', 'Junior (1-3 yrs)', 'Mid (3-6 yrs)', 
                                     'Senior (6-10 yrs)', 'Lead (10+ yrs)'], index=None)
    Openings = st.slider("Openings", 1, 20, 1)
    Applicants = st.slider("Applicants", 0, 2387, 100)
    Company_Rating = st.slider("Company Rating", 1.0, 5.0, 3.0)
    
    submit = st.form_submit_button("Predict Salary")

if submit:
    # Create the input data dictionary
    input_data = {
        "City": [City], "Company": [Company], "Job_Title": [Job_Title], "Industry": [Industry],
        "Job_Type": [Job_Type], "Education_Required": [Education_Required], "Location_Tier": [Location_Tier],
        "Work_Mode": [Work_Mode], "Company_Type": [Company_Type], "Experience_Level": [Experience_Level],
        "Openings": [Openings], "Applicants": [Applicants], "Company_Rating": [Company_Rating]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Apply same feature engineering as your original FastAPI code
    input_df['Applicants_per_Opening'] = input_df['Applicants'] / input_df['Openings']
    
    # Predict
    prediction = model.predict(input_df)
    st.success(f"Predicted Salary: {float(prediction[0]):.2f} LPA")