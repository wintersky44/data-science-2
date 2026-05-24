import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# 1. Load data and setup encoders (Must match your notebook exactly)
df = pd.read_csv("india_job_market_2024_2026.csv")
oe_cols = ['City', 'Company', 'Job_Title', 'Industry', 'Job_Type', 'Education_Required']
one_hot_cols = ['Location_Tier', 'Work_Mode', 'Company_Type']

# Initialize and fit encoders based on existing data
oe = OrdinalEncoder()
oe.fit(df[oe_cols])

experience_order = ['Fresher (0-1 yr)', 'Junior (1-3 yrs)', 'Mid (3-6 yrs)', 'Senior (6-10 yrs)', 'Lead (10+ yrs)']
exp_encoder = OrdinalEncoder(categories=[experience_order])
exp_encoder.fit(df[['Experience_Level']])

# 2. Train the model
X = pd.get_dummies(df.drop(columns=['Job_ID', 'Salary_LPA', 'Skills_Required', 'Date_Posted']), columns=one_hot_cols)
X[oe_cols] = oe.transform(df[oe_cols])
X['Experience_Level'] = exp_encoder.transform(df[['Experience_Level']])
y = df['Salary_LPA']

model = RandomForestRegressor(max_leaf_nodes=250, random_state=1)
model.fit(X, y)

# 3. Prediction Function
def predict_salary(city, company, job_title, industry, exp_level, job_type, edu, tier, work_mode, comp_type, rating, openings, applicants):
    # Create input dataframe
    input_data = pd.DataFrame({
        'Job_Title': [job_title], 'Company': [company], 'Industry': [industry], 'City': [city],
        'Experience_Level': [exp_level], 'Job_Type': [job_type], 'Education_Required': [edu],
        'Openings': [openings], 'Applicants': [applicants], 'Company_Rating': [rating],
        'Location_Tier': [tier], 'Work_Mode': [work_mode], 'Company_Type': [comp_type]
    })
    
    # Preprocess
    input_data[oe_cols] = oe.transform(input_data[oe_cols])
    input_data['Experience_Level'] = exp_encoder.transform(input_data[['Experience_Level']])
    input_data = pd.get_dummies(input_data, columns=one_hot_cols)
    
    # Reindex to match the training columns (handles missing dummy columns)
    input_data = input_data.reindex(columns=X.columns, fill_value=0)
    
    prediction = model.predict(input_data)
    return f"{round(prediction[0], 2)} LPA"

# 4. Define UI
demo = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Dropdown(df['City'].unique().tolist(), label="City"),
        gr.Dropdown(df['Company'].unique().tolist(), label="Company"),
        gr.Dropdown(df['Job_Title'].unique().tolist(), label="Job Title"),
        gr.Dropdown(df['Industry'].unique().tolist(), label="Industry"),
        gr.Dropdown(experience_order, label="Experience Level"),
        gr.Dropdown(df['Job_Type'].unique().tolist(), label="Job Type"),
        gr.Dropdown(df['Education_Required'].unique().tolist(), label="Education Required"),
        gr.Dropdown(['Tier 1', 'Tier 2', 'Remote'], label="Location Tier"),
        gr.Dropdown(['Hybrid', 'On-Site', 'Remote'], label="Work Mode"),
        gr.Dropdown(['Indian Unicorn', 'MNC', 'PSU/Govt', 'Startup'], label="Company Type"),
        gr.Slider(1, 5, value=3.5, label="Company Rating"),
        gr.Number(value=1, label="Openings"),
        gr.Number(value=100, label="Applicants")
    ],
    outputs="text",
    title="India Job Salary Predictor"
)

if __name__ == "__main__":
    demo.launch()