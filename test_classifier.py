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
MODELS_FOLDER = f"outputs_all_classes/inception_v3.tv_in1k/21_Jan/with_{num}_real" 
OUTPUT_EXCEL = f'excel/22_Jan_3_classes/results_{num}.xlsx'
model_name = "inception_v3.tv_in1k"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

test_dataset = GlaucomaHarvardDataset("test", model_name=model_name, augmentation=False, skip_early_glaucoma=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

classes_names = ['Advanced Glaucoma (0)', 'Early Glaucoma (1)', 'Healthy (2)']
labels_idx = [0, 1, 2]

# --- 2. Find all models ---
if not os.path.exists(MODELS_FOLDER):
    print(f"Error: Folder {MODELS_FOLDER} not found!")
    exit()

model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pth')]
model_files.sort() # Sort to ensure consistent order
print(f"Found {len(model_files)} models in {MODELS_FOLDER}")

# Container for all results
all_results_df = pd.DataFrame()

# --- 3. Iterate through models ---
for model_file in tqdm(model_files, desc="Evaluating Models"):
    model_path = os.path.join(MODELS_FOLDER, model_file)
    
    # Initialize Model
    model = GlaucomaClassifier(model_name=model_name)
    model.to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        if isinstance(state_dict, dict):
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
    
    # Confusion Matrix
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_idx)
    
    # Classification Report
    report = classification_report(
        actual_labels, 
        predicted_labels, 
        target_names=classes_names, 
        labels=labels_idx, 
        zero_division=0, 
        output_dict=True
    )

    # Construct Data Dictionary
    data_dict = {
        "model filename": model_file, 
        "overall accuracy": report['accuracy'], # type: ignore
        "------": "",
        "balanced accuracy %": (float(report[classes_names[0]]["recall"]) + float(report[classes_names[1]]["recall"])  + float(report[classes_names[2]]["recall"])) / 3,
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
        "----------": "",
    }

    # Convert to a Series and add to DataFrame
    current_col = pd.Series(data_dict, name=model_file)
    
    if all_results_df.empty:
        all_results_df = current_col.to_frame()
    else:
        all_results_df = pd.concat([all_results_df, current_col], axis=1)

# --- 4. Save to Excel ---
print("Saving results...")

if os.path.exists(OUTPUT_EXCEL):
    try:
        # Load existing
        existing_data = pd.read_excel(OUTPUT_EXCEL, index_col=0)
        print(f"Appending to existing {OUTPUT_EXCEL}...")
        
        # Merge columns (axis=1)
        combined_data = pd.concat([existing_data, all_results_df], axis=1)
        combined_data.to_excel(OUTPUT_EXCEL, index=True)
    except Exception as e:
        print(f"Could not read existing file (might be open?). Saving as new file instead. Error: {e}")
        all_results_df.to_excel("new_results_backup.xlsx", index=True)
else:
    # Create new
    all_results_df.to_excel(OUTPUT_EXCEL, index=True)

print(f"Done! Results saved to {OUTPUT_EXCEL}")