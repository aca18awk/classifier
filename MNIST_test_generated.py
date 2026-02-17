import glob
import os
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --- Configuration ---
EXPERIMENT_ROOT = "/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/models/15_Feb_Coloured_MNIST_FSFM_Latent/simulation_blue_GS_1_same_label"
MODEL_PATH = "outputs_MNIST/16_Feb_resnet18/distinctive-snowball-1_28_best_auc.pth"

# Derived Path
SAMPLES_DIR = os.path.join(EXPERIMENT_ROOT, "samples")

# --- HARDCODED GROUND TRUTH ---
GROUND_TRUTH_LABELS = [
    7,
    2,
    1,
    0,
    4,
    1,
    4,
    9,
    5,
    9,
    0,
    6,
    9,
    0,
    1,
    5,
    9,
    7,
    3,
    4,
    9,
    6,
    6,
    5,
    4,
    0,
    7,
    4,
    0,
    1,
    3,
    1,
    3,
    4,
    7,
    2,
    7,
    1,
    2,
    1,
    1,
    7,
    4,
    2,
    3,
    5,
    1,
    2,
    4,
    4,
]


# --- Model Definition ---
class Classifier(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=10, in_chans=1):
        super(Classifier, self).__init__()
        self.base_model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.base_model(x)


# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH}...")
model = Classifier(model_name="resnet18", num_classes=10, in_chans=1)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
else:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    exit()

model.to(device)
model.eval()

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def classify_image_digit(image_path):
    """
    Custom loader handling Blue text on Black background.
    Uses Max Channel projection to preserve brightness.
    """
    try:
        # Load as RGB
        img_rgb = Image.open(image_path).convert("RGB")
        img_np = np.array(img_rgb)

        # Max projection: Blue (0,0,255) -> 255 (White)
        img_max = np.max(img_np, axis=2)
        img = Image.fromarray(img_max.astype(np.uint8), mode="L")

        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.item()
    except Exception as e:
        print(f"Error classifying {image_path}: {e}")
        return None


def main():
    print("\n--- Starting Digit Analysis (With Error Indices) ---")
    print(f"Samples Directory: {SAMPLES_DIR}\n")

    if not os.path.exists(SAMPLES_DIR):
        print(f"Error: Samples directory not found at {SAMPLES_DIR}")
        return

    subfolders = [f.path for f in os.scandir(SAMPLES_DIR) if f.is_dir()]
    subfolders.sort(
        key=lambda f: int(os.path.basename(f)) if os.path.basename(f).isdigit() else f
    )

    total_images = 0
    total_correct = 0

    folder_stats = []

    for folder in subfolders:
        folder_name = os.path.basename(folder)

        # 1. Get Target
        try:
            folder_idx = int(folder_name)
            if 0 <= folder_idx < len(GROUND_TRUTH_LABELS):
                target_digit = GROUND_TRUTH_LABELS[folder_idx]
            else:
                continue
        except ValueError:
            continue

        # 2. Get Images
        image_files = glob.glob(os.path.join(folder, "*.png")) + glob.glob(
            os.path.join(folder, "*.jpg")
        )
        image_files = [
            f
            for f in image_files
            if "grid" not in f and "original" not in f and "reference" not in f
        ]

        if not image_files:
            continue

        # Use defaultdict(list) to store filenames per prediction
        # { 7: [], 2: ['img1', 'img5'] }
        predictions_map = defaultdict(list)

        # 3. Classify
        for img_path in image_files:
            predicted_digit = classify_image_digit(img_path)
            if predicted_digit is not None:
                # Extract simple ID from filename (e.g. "45" from "45.png")
                file_id = os.path.splitext(os.path.basename(img_path))[0]
                predictions_map[predicted_digit].append(file_id)

        # Calculate Stats
        count = len(image_files)
        # Count how many items are in the list corresponding to the target digit
        correct_list = predictions_map[target_digit]
        correct = len(correct_list)
        accuracy = (correct / count) * 100 if count > 0 else 0

        total_images += count
        total_correct += correct

        # 4. Build the Breakdown String
        # We want: "Target: Count, ErrorPred: [id1, id2]"
        breakdown_parts = []

        # Sort keys to ensure Correct Label comes first, then errors sorted by digit
        all_preds = sorted(predictions_map.keys(), key=lambda k: (k != target_digit, k))

        for p in all_preds:
            if p == target_digit:
                # For correct label, just show count
                breakdown_parts.append(f"{p}:{len(predictions_map[p])}")
            else:
                # For errors, show list of IDs
                # Sort IDs numerically if possible for neatness
                ids = sorted(
                    predictions_map[p], key=lambda x: int(x) if x.isdigit() else x
                )
                breakdown_parts.append(f"{p}:{ids}")

        folder_stats.append(
            {
                "name": folder_name,
                "target": target_digit,
                "count": count,
                "correct": correct,
                "acc": accuracy,
                "breakdown": ", ".join(breakdown_parts),
            }
        )

        print(f"Folder '{folder_name}' | Target: {target_digit} | Acc: {accuracy:.1f}%")

    # --- Final Summary Report ---
    print("\n" + "=" * 120)
    print(
        f"{'Folder':<8} | {'Target':<6} | {'Total':<8} | {'Correct':<8} | {'Accuracy':<10} | {'Breakdown (Correct:Count, Error:[IDs])'}"
    )
    print("-" * 120)

    for stat in folder_stats:
        print(
            f"{stat['name']:<8} | {stat['target']:<6} | {stat['count']:<8} | {stat['correct']:<8} | {stat['acc']:>6.1f}%   | {stat['breakdown']}"
        )

    print("-" * 120)

    if total_images > 0:
        total_acc = (total_correct / total_images) * 100
        print("OVERALL PERFORMANCE")
        print(f"Total Images Scanned: {total_images}")
        print(f"Total Correct: {total_correct}")
        print(f"Global Accuracy: {total_acc:.2f}%")
    else:
        print("No images found.")
    print("=" * 120)


if __name__ == "__main__":
    main()
