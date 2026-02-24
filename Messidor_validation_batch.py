import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from Messidor_class_generated import MessidorDataset


# --- 1. Define Model Architecture ---
class Classifier(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=5, in_chans=3):
        super(Classifier, self).__init__()
        self.base_model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.base_model(x)


if __name__ == "__main__":
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Evaluate Messidor models.")
    parser.add_argument(
        "--folder_path", type=str, default="Messidor_binary/24_Feb_hidden_weighted"
    )
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument(
        "--best_only",
        action="store_true",
        help="Only evaluate models that end in 'best_auc.pth'",
    )
    args = parser.parse_args()

    folder_path = args.folder_path

    # --- Determine if Binary or Multiclass ---
    # Normalizes the path (handles trailing slashes) and grabs the first directory
    path_parts = os.path.normpath(folder_path).split(os.sep)
    first_folder = path_parts[0].lower()

    is_binary = "binary" in first_folder
    num_classes = 2 if is_binary else 5

    print("--- Initialization ---")
    print(f"Target Folder: {folder_path}")
    print(
        f"Detected Mode: {'Binary (2-Class)' if is_binary else 'Multiclass (5-Class)'}"
    )

    file_path = os.path.join(folder_path, "evaluation_results.xlsx")

    # Find .pth files
    if os.path.exists(folder_path):
        paths = []
        for f in os.listdir(folder_path):
            if args.best_only:
                if f.endswith("best_auc.pth"):
                    paths.append(os.path.join(folder_path, f))
            else:
                if f.endswith(".pth"):
                    paths.append(os.path.join(folder_path, f))

        paths.sort()
        print(f"Found {len(paths)} model(s) to evaluate.")
    else:
        raise FileNotFoundError(
            f"CRITICAL: folder_path '{folder_path}' does not exist."
        )

    if len(paths) == 0:
        print("No .pth files found. Exiting.")
        exit()

    IMG_SIZE = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- 2. Load Model & Data ---
    model = Classifier(model_name=args.model_name, num_classes=num_classes, in_chans=3)
    model.to(device)

    test_dataset = MessidorDataset("test", img_size=IMG_SIZE, is_binary=is_binary)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    results_list = []

    for model_path in paths:
        print(f"Evaluating: {os.path.basename(model_path)}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # --- 3. Inference ---
        predicted_labels = []
        predicted_probs = []
        actual_labels = []

        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="Inference", leave=False):
                images = images.to(device)
                outputs = model(images)

                # Get hard predictions
                _, preds = torch.max(outputs, 1)

                if is_binary:
                    # Extract probability of the positive class (Class 1) for AUC
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    predicted_probs.extend(probs.cpu().numpy())

                predicted_labels.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())

        predicted_labels = np.array(predicted_labels)
        actual_labels = np.array(actual_labels)
        if is_binary:
            predicted_probs = np.array(predicted_probs)

        # --- 4. Metrics & Reporting ---
        report = classification_report(
            actual_labels, predicted_labels, output_dict=True, zero_division=0
        )

        balanced_acc = balanced_accuracy_score(actual_labels, predicted_labels)
        overall_acc = accuracy_score(actual_labels, predicted_labels)
        macro_f1 = report["macro avg"]["f1-score"]  # type: ignore

        data_dict = {"Metric": []}
        values = []

        if is_binary:
            # --- Binary Specific Metrics ---
            try:
                auc = roc_auc_score(actual_labels, predicted_probs)
            except ValueError:
                auc = 0.0

            specificity = report["0"]["recall"] if "0" in report else -1  # type: ignore
            sensitivity = report["1"]["recall"] if "1" in report else -1  # type: ignore

            data_dict["Metric"] = [
                "ROC-AUC",
                "Overall Accuracy",
                "Balanced Accuracy",
                "Macro F1-Score",
                "Sensitivity (Class 1 Recall)",
                "Specificity (Class 0 Recall)",
            ]
            values = [
                auc,
                overall_acc,
                balanced_acc,
                macro_f1,
                sensitivity,
                specificity,
            ]

            print(
                f"Results -> AUC: {auc:.4f} | Balanced Acc: {balanced_acc:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f}\n"
            )

        else:
            # --- Multiclass Specific Metrics ---
            qwk = cohen_kappa_score(
                actual_labels, predicted_labels, weights="quadratic"
            )

            data_dict["Metric"] = [
                "QWK (Quadratic Kappa)",
                "Overall Accuracy",
                "Balanced Accuracy",
                "Macro F1-Score",
            ]
            values = [qwk, overall_acc, balanced_acc, macro_f1]

            for i in range(num_classes):
                class_key = str(i)
                recall = report[class_key]["recall"] if class_key in report else -1  # type: ignore
                data_dict["Metric"].append(f"Class {i} Recall (Sensitivity)")
                values.append(recall)

            print(f"Results -> QWK: {qwk:.4f} | Balanced Acc: {balanced_acc:.4f}\n")

        # --- 5. Package for Excel ---
        model_col_name = os.path.basename(model_path)
        current_model_df = pd.DataFrame(
            values, index=data_dict["Metric"], columns=[model_col_name]
        )
        results_list.append(current_model_df)

    # --- 6. Save to Excel ---
    if results_list:
        final_df = pd.concat(results_list, axis=1)
        print("--- Final Summary Table ---")
        print(final_df)

        print(f"\nSaving to {file_path}...")
        if os.path.exists(file_path):
            try:
                existing_data = pd.read_excel(file_path, index_col=0)
                combined_data = pd.concat([existing_data, final_df], axis=1)
                # Remove duplicate columns if evaluating the same model twice
                combined_data = combined_data.loc[
                    :, ~combined_data.columns.duplicated()
                ]
                combined_data.to_excel(file_path)
            except Exception as e:
                print(
                    f"Error appending to existing excel: {e}. Overwriting with new file."
                )
                final_df.to_excel(file_path)
        else:
            final_df.to_excel(file_path)

    print("\nDone.")
