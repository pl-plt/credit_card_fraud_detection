import kagglehub
import shutil

custom_path = "../dataset/raw"

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)


shutil.move(path, custom_path)
print("Dataset moved to:", custom_path)