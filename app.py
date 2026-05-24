import streamlit as st
import requests

st.title("Job Salary Predictor")

import json

with open('categories.json', 'r') as f:
    categories = json.load(f)

# Input form
with st.form("salary_form"):
    City = st.selectbox("City", options=categories['cities'],index=None, placeholder="Select a city...")
    Company = st.selectbox("Company", options=categories['companies'],index=None, placeholder="Select a company...")
    Job_Title = st.selectbox("Job Title", options=categories['job titles'],index=None, placeholder="Select a Job Title...")
    Industry = st.selectbox("Industry", options=categories['industries'],index=None, placeholder="Select an industry...")
    Job_Type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract"],index=None, placeholder="Select a job type...")
    Education_Required = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"],index=None, placeholder="Select education level...")
    Location_Tier = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"],index=None, placeholder="Select location tier...")
    Work_Mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "Office"],index=None, placeholder="Select work mode...")
    Company_Type = st.selectbox("Company Type", ["Startup", "MNC", "SME"],index=None, placeholder="Select company type...")
    Experience_Level = st.selectbox("Experience Level", 
                                    ['Fresher (0-1 yr)', 'Junior (1-3 yrs)', 'Mid (3-6 yrs)', 
                                     'Senior (6-10 yrs)', 'Lead (10+ yrs)'],index=None, placeholder="Select experience level...")
    Openings = st.slider("Openings", min_value=1, max_value=20, value=1)
    Applicants = st.slider("Applicants", min_value=0, max_value=2387, value=100)
    Company_Rating = st.slider("Company Rating", 1.0, 5.0, 3.0)
    
    submit = st.form_submit_button("Predict Salary")

if submit:
    # Send data to FastAPI
    payload = {
        "City": City, "Company": Company, "Job_Title": Job_Title, "Industry": Industry,
        "Job_Type": Job_Type, "Education_Required": Education_Required, "Location_Tier": Location_Tier,
        "Work_Mode": Work_Mode, "Company_Type": Company_Type, "Experience_Level": Experience_Level,
        "Openings": Openings, "Applicants": Applicants, "Company_Rating": Company_Rating
    }
    
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    
    if response.status_code == 200:
        salary = response.json()['predicted_salary_lpa']
        st.success(f"Predicted Salary: {salary:.2f} LPA")
    else:
        st.error("Error connecting to prediction service.")