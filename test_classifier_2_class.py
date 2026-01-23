import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm 
import argparse


# --- Import your custom modules ---
from custom_model import GlaucomaClassifier
from dataset import GlaucomaHarvardDataset

parser = argparse.ArgumentParser()
parser.add_argument("--num", default=10)
args = parser.parse_args()
config = vars(args)

num = int(config["num"])

# --- Configuration ---
# MODELS_FOLDER = f"outputs_limited_data/inception_v3.tv_in1k/7_Jan_baseline_20_with_flow_{num}/assess" 
MODELS_FOLDER = f"outputs_with_early_glaucoma/inception_v3.tv_in1k/20_Jan/with_all_real"
OUTPUT_EXCEL = f'excel/20_Jan/early_glaucoma/results_all_shuffle.xlsx'
model_name = "inception_v3.tv_in1k"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

test_dataset = GlaucomaHarvardDataset(
    "test", 
    model_name=model_name, 
    augmentation=False, 
    skip_early_glaucoma=True
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# 2. Define 2-Class Labels (Mapped: 0=Advanced, 1=Normal)
classes_names = ['Early Glaucoma', 'Normal Control']
labels_idx = [0, 1]

# --- 3. Find all models ---
if not os.path.exists(MODELS_FOLDER):
    print(f"Error: Folder {MODELS_FOLDER} not found!")
    exit()

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)

model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pth')]
model_files.sort()
print(f"Found {len(model_files)} models in {MODELS_FOLDER}")

# Container for all results
all_results_df = pd.DataFrame()

# --- 4. Iterate through models ---
for model_file in tqdm(model_files, desc="Evaluating Models"):
    model_path = os.path.join(MODELS_FOLDER, model_file)
    
    # Initialize Model with num_classes=2
    model = GlaucomaClassifier(model_name=model_name, num_classes=2)
    model.to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        if isinstance(state_dict, dict):
            # Handle if state_dict is nested (e.g. checkpoint saving optimizer state too)
            if 'model_state_dict' in state_dict:
                 model.load_state_dict(state_dict['model_state_dict'])
            else:
                 model.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict.state_dict())
            
    except Exception as e:
        print(f"Skipping {model_file}: Could not load weights. Error: {e}")
        continue
        
    model.eval()

    # Inference Loop
    predicted_labels = []
    actual_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            
            example_out = model(images)
            probabilities = torch.nn.functional.softmax(example_out, dim=1) 
            class_id = [p.argmax().item() for p in probabilities]
            
            predicted_labels.extend(class_id)
            actual_labels.extend(labels.tolist())

    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array(actual_labels)

    # --- Metrics Calculation ---
    
    # Confusion Matrix (2x2)
    # [0,0] Adv->Adv (TP), [0,1] Adv->Norm (FN)
    # [1,0] Norm->Adv (FP), [1,1] Norm->Norm (TN)
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_idx)
    
    # Check shape to be safe (in case a batch missed a class entirely)
    if cm.shape != (2, 2):
        # Force it to 2x2 if sklearn reduces it
        cm_full = np.zeros((2, 2), dtype=int)
        # (This is a simplified safety check, usually not needed if labels_idx is passed)
        cm = confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])

    # Classification Report
    report = classification_report(
        actual_labels, 
        predicted_labels, 
        target_names=classes_names, 
        labels=labels_idx, 
        zero_division=0, 
        output_dict=True
    )

    # Construct Data Dictionary for 2 Classes
    data_dict = {
        # "model filename": model_file, 
        "overall accuracy": report['accuracy'],
        "------": "",
        "recall (Early glaucoma) %": report[classes_names[0]]["recall"],
        "recall (normal control) %": report[classes_names[1]]["recall"],
        "-------": "",
        # CM Layout for 2 Classes
        "Early -> Early (Correct)": cm[0,0],
        "Early -> Normal (Miss)":      cm[0,1],
        "Normal -> Normal (Correct)":     cm[1,1],
        "Normal -> Early (False Alarm)": cm[1,0],
        "--------": "",
        "f1 (Early glaucoma)": report[classes_names[0]]["f1-score"],
        "f1 (normal control)":    report[classes_names[1]]["f1-score"],
        "----------": "",
        "precision (Early glaucoma)": report[classes_names[0]]["precision"],
        "precision (normal control)":    report[classes_names[1]]["precision"],
        "----------": "",
        "balanced accuracy %": (float(report[classes_names[0]]["recall"]) + float(report[classes_names[1]]["recall"])) / 2,
    }

    # Convert to a Series and add to DataFrame
    current_col = pd.Series(data_dict, name=model_file)
    
    if all_results_df.empty:
        all_results_df = current_col.to_frame()
    else:
        all_results_df = pd.concat([all_results_df, current_col], axis=1)

# --- 5. Save to Excel ---
print("Saving results...")

if os.path.exists(OUTPUT_EXCEL):
    try:
        existing_data = pd.read_excel(OUTPUT_EXCEL, index_col=0)
        print(f"Appending to existing {OUTPUT_EXCEL}...")
        combined_data = pd.concat([existing_data, all_results_df], axis=1)
        combined_data.to_excel(OUTPUT_EXCEL, index=True)
    except Exception as e:
        print(f"Could not read existing file. Saving as new file 'new_results_backup.xlsx'. Error: {e}")
        all_results_df.to_excel("new_results_backup.xlsx", index=True)
else:
    all_results_df.to_excel(OUTPUT_EXCEL, index=True)

print(f"Done! Results saved to {OUTPUT_EXCEL}")