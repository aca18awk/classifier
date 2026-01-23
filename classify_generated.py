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
model_path = "outputs2/inception_v3.tv_in1k/26_Nov_gen_data_experiments/vocal-fire-67_51_data_0_best_auc.pth"
file_path = '18_12_generated_data.xlsx'
data_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/18_Dec/fluent-morning-29_GS_20.0'

# We need to extract the architecture name (e.g. "inception_v3.tv_in1k") 
# so the Model class knows what to build and the Dataset knows how to resize.
# Based on your path structure, it is the parent folder name or the first part of the filename.
# Let's hardcode it to be safe, or extract it:
model_name_arch = "inception_v3.tv_in1k" 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load Model ---
# We must tell the class WHICH architecture to build before loading weights
model = GlaucomaClassifier(model_name=model_name_arch)
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
predicted_labels = []
actual_labels = []

print("Running inference...")
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        
        example_out = model(images)
        probabilities = torch.nn.functional.softmax(example_out, dim=1) 
        class_id = [p.argmax().item() for p in probabilities]
        
        predicted_labels.extend(class_id)
        actual_labels.extend(labels.tolist())

predicted_labels = np.array(predicted_labels)
actual_labels = np.array(actual_labels)

# --- 4. Metrics ---
classes_names = ['Advanced Glaucoma (0)', 'Early Glaucoma (1)', 'Healthy (2)']
labels_idx = [0, 1, 2]

# Rename variable to avoid shadowing the function import
cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_idx)
report = classification_report(actual_labels, predicted_labels, target_names=classes_names, labels=labels_idx, zero_division=0, output_dict=True)

# Build the data dictionary
data_dict = {
    "model path": model_path,
    "overall accuracy": report['accuracy'], # type: ignore
    "------": "",
    "recall (advanced glaucoma) %": report[classes_names[0]]["recall"], # type: ignore
    "recall (early glaucoma) %": report[classes_names[1]]["recall"], # type: ignore
    "recall (healthy) %": report[classes_names[2]]["recall"], # type: ignore
    "-------": "",
    # CM Layout: Rows = True, Cols = Predicted
    "advanced glaucoma -> advanced glaucoma": cm[0,0],
    "advanced glaucoma -> healthy": cm[0,2],       # DANGEROUS MISS
    "advanced glaucoma -> early glaucoma": cm[0,1],
    "early glaucoma -> early glaucoma": cm[1,1],
    "early glaucoma -> healthy": cm[1,2],
    "early glaucoma -> advanced glaucoma": cm[1,0],
    "healthy -> healthy": cm[2,2],
    "healthy -> advanced glaucoma": cm[2,0],       # FALSE POSITIVE
    "healthy -> early glaucoma": cm[2,1],
    "--------": "",
    "f1 (advanced glaucoma)": report[classes_names[0]]["f1-score"], # type: ignore
    "f1 (early glaucoma)": report[classes_names[1]]["f1-score"], # type: ignore
    "f1 (healthy)": report[classes_names[2]]["f1-score"], # type: ignore
    "----------": "",
    "precision (advanced glaucoma)": report[classes_names[0]]["precision"], # type: ignore
    "precision (early glaucoma)": report[classes_names[1]]["precision"], # type: ignore
    "precision (healthy)": report[classes_names[2]]["precision"], # type: ignore
    }

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