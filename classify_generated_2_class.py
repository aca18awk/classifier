import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# --- Import your custom modules ---
from custom_model import GlaucomaClassifier
from generated_dataset import GeneratedGlaucomaDataset

# --- Configuration ---
model_path = "outputs_limited_data/inception_v3.tv_in1k/7_Jan_baseline_200/hopeful-cherry-12_12_data_0_best_loss.pth"
file_path = '20_01_generated_data_flow_euler_20.xlsx'
data_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/19_Jan_flow_euler/zesty-fire-39-best_GS_20.0'
# data_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/Jan_15_diff_conditional_model_90000_DDPM_2'
# data_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/18_Dec/fluent-morning-29_GS_20.0'

# We need to extract the architecture name (e.g. "inception_v3.tv_in1k") 
# so the Model class knows what to build and the Dataset knows how to resize.
# Based on your path structure, it is the parent folder name or the first part of the filename.
# Let's hardcode it to be safe, or extract it:
model_name_arch = "inception_v3.tv_in1k" 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model ---
# We must tell the class WHICH architecture to build before loading weights
model = GlaucomaClassifier(model_name=model_name_arch, num_classes=2)
model.to(device)

print(f"Loading weights from {model_path}...")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --- 2. Load Data ---
# Note: Removed manual transforms. The dataset class handles resizing based on model_name.
test_dataset = GeneratedGlaucomaDataset(path=data_path, model_name=model_name_arch)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 3. Inference ---
# --- 3. Inference ---
predicted_labels = []
actual_labels = []

print("Running inference...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        # For 2 classes, argmax still works perfectly
        preds = torch.argmax(outputs, dim=1)
        
        predicted_labels.extend(preds.cpu().numpy())
        actual_labels.extend(labels.numpy())

# --- 4. Metrics (Adjusted for 2 Classes) ---
classes_names = ['Advanced Glaucoma (0)', 'Healthy (1)']
labels_idx = [0, 1]

cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_idx)
report = classification_report(actual_labels, predicted_labels, target_names=classes_names, labels=labels_idx, zero_division=0, output_dict=True)

# Build the data dictionary for 2 classes
data_dict = {
    "model path": model_path,
    "overall accuracy": report['accuracy'],
    "------": "",
    "recall (advanced) %": report[classes_names[0]]["recall"],
    "recall (healthy) %": report[classes_names[1]]["recall"],
    "-------": "",
    "adv -> adv (Correct)": cm[0,0],
    "adv -> healthy (Dangerous Miss)": cm[0,1],
    "healthy -> healthy (Correct)": cm[1,1],
    "healthy -> adv (False Positive)": cm[1,0],
    "--------": "",
    "f1 (advanced)": report[classes_names[0]]["f1-score"],
    "f1 (healthy)": report[classes_names[1]]["f1-score"],
    "----------": "",
    "precision (advanced)": report[classes_names[0]]["precision"],
    "precision (healthy)": report[classes_names[1]]["precision"],
}

# ... rest of your pandas saving logic remains the same ...

# Create a single column DataFrame
# The column name will be the model filename
col_name = model_path.split('/')[-1]
new_column = pd.Series(data_dict, name=col_name)
new_df = new_column.to_frame()

# --- 5. Save to Excel ---
print(f"Saving to {file_path}...")

if os.path.exists(file_path):
    try:
        # Load existing sheet
        existing_data = pd.read_excel(file_path, index_col=0)
        # Merge columns
        combined_data = pd.concat([existing_data, new_df], axis=1)
        combined_data.to_excel(file_path, index=True)
    except Exception as e:
        print(f"Error reading existing excel: {e}. Saving new file.")
        new_df.to_excel(file_path, index=True)
else:
    new_df.to_excel(file_path, index=True)

print("Done.")