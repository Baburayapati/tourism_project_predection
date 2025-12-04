from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN") or "os.environ['HF_TOKEN']"

api = HfApi(token=HF_TOKEN)

api.upload_folder(
folder_path="tourism_project/deployment",     # the local folder containing your Streamlit app files
repo_id="BabuRayapati/tourism_project_app",  # your target Hugging Face Space repo
repo_type="space",                         # dataset, model, or space
path_in_repo="",                            # optional: subfolder path inside the repo
)

print("Deployment folder uploaded successfully!")
