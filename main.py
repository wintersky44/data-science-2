from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the model
model = joblib.load('model.joblib')

class JobDetails(BaseModel):
    City: str
    Company: str
    Job_Title: str
    Industry: str
    Job_Type: str
    Education_Required: str
    Location_Tier: str
    Work_Mode: str
    Company_Type: str
    Experience_Level: str
    Openings: int
    Applicants: int
    Company_Rating: float

@app.post("/predict")
def predict_salary(job: JobDetails):
    # Convert input to DataFrame
    input_df = pd.DataFrame([job.dict()])
    
    # Feature Engineering (must match notebook logic)
    input_df['Applicants_per_Opening'] = input_df['Applicants'] / input_df['Openings']
    
    # Predict
    prediction = model.predict(input_df)
    return {"predicted_salary_lpa": float(prediction[0])}