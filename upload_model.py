from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="Backend/bert_ged_model",
    repo_id="swayamt/grammar-error-detection",
    repo_type="model"
)

print("Done!")