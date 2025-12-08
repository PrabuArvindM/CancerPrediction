import os
import shutil

# Source folder (change path if needed)
source_folder = "IDC_regular_ps50_idx5"

# Destination folders
cancer_dir = "data/cancer"
non_cancer_dir = "data/non_cancer"

# Create destination folders if not exist
os.makedirs(cancer_dir, exist_ok=True)
os.makedirs(non_cancer_dir, exist_ok=True)

count_cancer = 0
count_non_cancer = 0

# Go through all subfolders
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):
            full_path = os.path.join(root, file)
            
            # If the folder name ends with '/1', it's cancer
            if os.path.basename(os.path.dirname(full_path)) == "1":
                dest = os.path.join(cancer_dir, file)
                shutil.copy(full_path, dest)
                count_cancer += 1
            else:
                dest = os.path.join(non_cancer_dir, file)
                shutil.copy(full_path, dest)
                count_non_cancer += 1

print(f"✅ Done! Copied {count_cancer} cancer and {count_non_cancer} non-cancer images.")
 