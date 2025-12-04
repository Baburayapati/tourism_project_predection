import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model from your HuggingFace repo

model_path = hf_hub_download(
repo_id="BabuRayapati/tourism_project_model",
filename="best_tourism_project_model_v1.joblib"
)
model = joblib.load(model_path)

st.title("Tourism Product Purchase Prediction App")
st.write("""
This app predicts the likelihood or value of a customer taking a product based on their profile and travel history.
Please enter the customer details below to get a prediction.
""")

# User inputs

Age = st.number_input("Age", min_value=18, max_value=100, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=30, value=10)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1,2,3,4,5])
NumberOfTrips = st.number_input("Number of Previous Trips", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Has Passport", [0, 1])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=10, value=8)
OwnCar = st.selectbox("Owns Car", [0,1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)

# Categorical features

TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Business", "Student", "Other"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Premium"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Staff", "Other"])

# Assemble input into DataFrame

input_data = pd.DataFrame([{
'Age': Age,
'CityTier': CityTier,
'DurationOfPitch': DurationOfPitch,
'NumberOfPersonVisiting': NumberOfPersonVisiting,
'NumberOfFollowups': NumberOfFollowups,
'PreferredPropertyStar': PreferredPropertyStar,
'NumberOfTrips': NumberOfTrips,
'Passport': Passport,
'PitchSatisfactionScore': PitchSatisfactionScore,
'OwnCar': OwnCar,
'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
'MonthlyIncome': MonthlyIncome,
'TypeofContact': TypeofContact,
'Occupation': Occupation,
'Gender': Gender,
'ProductPitched': ProductPitched,
'MaritalStatus': MaritalStatus,
'Designation': Designation
}])

if st.button("Predict Product Taken"):
prediction = model.predict(input_data)[0]
st.subheader("Prediction Result:")
st.success(f"The model predicts ProdTaken value: **{prediction:.2f}**")
