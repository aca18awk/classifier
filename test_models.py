import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm # For progress bar

# --- Import your custom modules ---
from custom_model import GlaucomaClassifier
from dataset import GlaucomaHarvardDataset

# --- Configuration ---
MODELS_FOLDER = "outputs/inception_v3.tv_in1k/25_Nov_gen_data_2"
OUTPUT_EXCEL = 'augmented_batch_results5.xlsx'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Setup Data (Do this once to save time) ---
print("Initializing dataset and dataloader...")
transform1 = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
    )
])

# Assuming GlaucomaHarvardDataset is defined/imported correctly
test_dataset = GlaucomaHarvardDataset("test", transform1, augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

classes_names = ['Advanced Glaucoma (0)', 'Early Glaucoma (1)', 'Healthy (2)']
labels_idx = [0, 1, 2]

# --- 2. Find all models ---
model_files = [f for f in os.listdir(MODELS_FOLDER) if f.endswith('.pth')]
model_files.sort() # Sort to ensure consistent order
print(f"Found {len(model_files)} models in {MODELS_FOLDER}")

# Container for all results
all_results_df = pd.DataFrame()

# --- 3. Iterate through models ---
for model_file in tqdm(model_files, desc="Evaluating Models"):
    model_path = os.path.join(MODELS_FOLDER, model_file)
    
    # Initialize and Load Model
    # We re-initialize or reload strictly to ensure no weight leakage between iterations
    model = GlaucomaClassifier(model_name="inception_v3.tv_in1k")
    model.to(DEVICE)
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading {model_file}: {e}")
        continue
        
    model.eval()

    # Inference
    predicted_labels = []
    actual_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            # labels = labels.to(DEVICE) # Not needed on GPU for metric accumulation
            
            example_out = model(images)
            probabilities = torch.nn.functional.softmax(example_out, dim=1) 
            class_id = [p.argmax().item() for p in probabilities]
            
            predicted_labels.extend(class_id)
            actual_labels.extend(labels.tolist())

    predicted_labels = np.array(predicted_labels)
    actual_labels = np.array(actual_labels)

    # Metrics
    cm = confusion_matrix(actual_labels, predicted_labels, labels=labels_idx)
    report = classification_report(actual_labels, predicted_labels, target_names=classes_names, labels=labels_idx, zero_division=0, output_dict=True)

    # Construct Data Dictionary for this specific model
    # Note: We create a single column DataFrame here to match your desired structure
    data_dict = {
        "model path": model_file, # Just filename is usually cleaner, or use model_path
        "overall accuracy": report['accuracy'],
        "------": "",
        "recall (advanced glaucoma) %": report[classes_names[0]]["recall"],
        "recall (early glaucoma) %": report[classes_names[1]]["recall"],
        "recall (healthy) %": report[classes_names[2]]["recall"],
        "-------": "",
        "advanced glaucoma -> advanced glaucoma": cm[0,0],
        "advanced glaucoma -> healthy": cm[0,2],
        "advanced glaucoma -> early glaucoma": cm[0,1],
        "early glaucoma -> early glaucoma": cm[1,1],
        "early glaucoma -> healthy": cm[1,2],
        "early glaucoma -> advanced glaucoma": cm[1,0],
        "healthy -> healthy": cm[2,2],
        "healthy -> advanced glaucoma": cm[2,0],
        "healthy -> early glaucoma": cm[2,1],
        "--------": "",
        "f1 (advanced glaucoma)": report[classes_names[0]]["f1-score"],
        "f1 (early glaucoma)": report[classes_names[1]]["f1-score"],
        "f1 (healthy)": report[classes_names[2]]["f1-score"],
        "----------": "",
        "precision (advanced glaucoma)": report[classes_names[0]]["precision"],
        "precision (early glaucoma)": report[classes_names[1]]["precision"],
        "precision (healthy)": report[classes_names[2]]["precision"],
    }

    # Convert to a Series (column) and give it the model name
    current_col = pd.Series(data_dict, name=model_file)
    
    # Append to main dataframe
    if all_results_df.empty:
        all_results_df = current_col.to_frame()
    else:
        all_results_df = pd.concat([all_results_df, current_col], axis=1)

# --- 4. Save to Excel ---
print("Saving results...")

if os.path.exists(OUTPUT_EXCEL):
    # If file exists, load it and append new columns
    existing_data = pd.read_excel(OUTPUT_EXCEL, index_col=0)
    # Align indices just in case
    combined_data = pd.concat([existing_data, all_results_df], axis=1)
    combined_data.to_excel(OUTPUT_EXCEL, index=True)
else:
    # Create new
    all_results_df.to_excel(OUTPUT_EXCEL, index=True)

print(f"Done! Results saved to {OUTPUT_EXCEL}")