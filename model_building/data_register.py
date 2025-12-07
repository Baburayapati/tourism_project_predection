from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os
import glob

repo_id = "BabuRayapati/tourism_project"
repo_type = "dataset"

# Correct token handling
HF_TOKEN = os.getenv("HF_TOKEN") or "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"
api = HfApi(token=HF_TOKEN)

# Step 1: Check if dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload all files from folder
folder_path = "tourism_project/data"  # Folder containing CSVs
files = glob.glob(f"{folder_path}/*")  # Get all files in folder

for file_path in files:
    file_name = os.path.basename(file_path)
    print(f"Uploading {file_name}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type=repo_type,
    )

print("All files uploaded successfully!")
