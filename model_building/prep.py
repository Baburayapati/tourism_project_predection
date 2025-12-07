import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import Repository
import shutil

# --- Step 0: HF token ---
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Step 1: Load dataset ---
DATASET_PATH = "https://huggingface.co/datasets/BabuRayapati/tourism_project/raw/main/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop UDI if exists
if "UDI" in df.columns:
    df.drop(columns=["UDI"], inplace=True)

# Encode categorical columns
if "Type" in df.columns:
    label_encoder = LabelEncoder()
    df["Type"] = label_encoder.fit_transform(df["Type"])

# --- Step 2: Split data ---
target_col = "Failure"
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Save CSV locally ---
os.makedirs("temp_csvs", exist_ok=True)
Xtrain.to_csv("temp_csvs/Xtrain.csv", index=False)
Xtest.to_csv("temp_csvs/Xtest.csv", index=False)
ytrain.to_csv("temp_csvs/ytrain.csv", index=False)
ytest.to_csv("temp_csvs/ytest.csv", index=False)
print("Local CSV files created successfully.")

# --- Step 4: Upload to HF Dataset Repo using Repository ---
repo_id = "BabuRayapati/tourism_project"
repo_url = f"https://huggingface.co/datasets/{repo_id}"
repo_local_path = "temp_repo"

# Clone the repo locally
repo = Repository(local_dir=repo_local_path, clone_from=repo_url, use_auth_token=HF_TOKEN)

# Copy CSVs to repo folder
shutil.copytree("temp_csvs", repo_local_path, dirs_exist_ok=True)

# Commit and push
repo.push_to_hub(commit_message="Upload processed CSV files")
print("All files uploaded successfully to HuggingFace dataset repo!")
