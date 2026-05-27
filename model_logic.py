import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("india_job_market_2024_2026.csv")

oe_cols = ['City', 'Company', 'Job_Title', 'Industry', 'Job_Type', 'Education_Required']
one_hot_cols = ['Location_Tier', 'Work_Mode', 'Company_Type']
exp_col = ['Experience_Level']
passthrough_cols = ['Openings', 'Applicants', 'Company_Rating', 'Applicants_per_Opening']
cols_to_drop = ['Job_ID', 'Salary_LPA', 'Skills_Required', 'Date_Posted']

df['Applicants_per_Opening'] = df['Applicants'] / df['Openings']

experience_order = [
    'Fresher (0-1 yr)', 'Junior (1-3 yrs)', 'Mid (3-6 yrs)', 
    'Senior (6-10 yrs)', 'Lead (10+ yrs)'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('oe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), oe_cols),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), one_hot_cols),
        ('exp', OrdinalEncoder(categories=[experience_order]), exp_col),
        ('passthrough', 'passthrough', passthrough_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(max_leaf_nodes=250, random_state=1))
])

X = df.drop(columns=cols_to_drop)
y = df['Salary_LPA']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

model_pipeline.fit(train_X, train_y)

joblib.dump(model_pipeline, 'model.joblib')

