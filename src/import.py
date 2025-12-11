import kagglehub
import shutil
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download Credit Card Fraud Detection dataset")
    parser.add_argument("--output", type=str, default="dataset/raw", help="Output directory for the dataset")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Check if dataset already exists (simple check for the main file)
    expected_file = os.path.join(args.output, "creditcard.csv")
    if os.path.exists(expected_file):
        print(f"Dataset already exists at {expected_file}. Skipping download.")
        return

    print("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print("Path to downloaded files:", path)

        # Move files to target directory
        print(f"Moving files to {args.output}...")
        for filename in os.listdir(path):
            src_file = os.path.join(path, filename)
            dst_file = os.path.join(args.output, filename)
            if os.path.isfile(src_file):
                shutil.move(src_file, dst_file)
        
        print("Dataset download and setup complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        exit(1)

if __name__ == "__main__":
    main()