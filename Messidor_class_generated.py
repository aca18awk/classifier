import os
from typing import Literal

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset import GammaCorrectionTransform

# Added "generated" to the valid options
MessidorSplit = Literal["hospital_b", "test", "hidden", "validation", "generated"]


class MessidorDataset(Dataset):
    def __init__(
        self,
        purpose: MessidorSplit = "test",
        root_dir: str = "/vol/biomedic3/awk24/datasets/Messidor2_256",
        csv_path: str = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv",
        img_size=128,
        generated_dir=None,  # New parameter for the generated samples path
        is_binary=False,
    ):
        """
        Args:
            purpose: One of "hospital_b", "test", "hidden", "validation", or "generated".
            root_dir: Path to the parent folder containing the processed subfolders.
            csv_path: Path to the original CSV with labels.
            generated_dir: Path to the 'samples' folder containing generated images.
        """
        self.purpose = purpose
        self.image_paths = []
        self.labels = {}
        self.img_size = img_size
        self.is_binary = is_binary

        # 1. Select Subfolder based on Purpose
        if purpose == "hospital_b":
            self.data_dir = os.path.join(root_dir, "hospital_b")
        elif purpose == "test":
            self.data_dir = os.path.join(root_dir, "test")
        elif purpose == "hidden":
            self.data_dir = os.path.join(root_dir, "hidden_classifier_data")
        elif purpose == "validation":
            self.data_dir = os.path.join(root_dir, "validation")
        elif purpose == "generated":
            if generated_dir is None:
                raise ValueError(
                    "You must provide `generated_dir` when using purpose='generated'."
                )
            self.data_dir = generated_dir
        else:
            raise ValueError(
                f"Invalid purpose '{purpose}'. Must be 'hospital_b', 'test', 'hidden', 'validation', or 'generated'."
            )

        # 2. Load CSV Data (Labels)
        if csv_path and os.path.exists(csv_path):
            print(f"Loading labels from {csv_path}...")
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                # We assume 'id_code' matches the filename stem (e.g. IM0001)
                key = str(row["id_code"]).strip()
                self.labels[key] = row.to_dict()
        else:
            print(f"Warning: CSV path {csv_path} not found. Labels will be -1.")

        # 3. Collect Images from the Specific Subfolder
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}
        print(f"Scanning files in {self.data_dir}...")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"Directory {self.data_dir} does not exist. Please verify the path."
            )

        # os.walk naturally dives into all subfolders (e.g., 0_20051020_45050_0100_PP)
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_extensions:
                    # Ignore grid images if they exist in the sample folders
                    if "grid" not in file:
                        self.image_paths.append(os.path.join(root, file))

        self.image_paths.sort()
        print(f"Found {len(self.image_paths)} images for split '{purpose}'.")

        # 4. Define Transform
        self.transform_train = transforms.Compose(
            [
                transforms.Resize([img_size, img_size]),
                transforms.RandomApply(
                    transforms=[GammaCorrectionTransform(gamma=0.3)], p=0.5
                ),
                transforms.RandomApply(
                    transforms=[transforms.ColorJitter(brightness=0.3, contrast=0.3)],
                    p=0.5,
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
                transforms.RandomApply(
                    transforms=[transforms.RandomPerspective(distortion_scale=0.2)],
                    p=0.5,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize([img_size, img_size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        filename = os.path.basename(img_path)

        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new("RGB", (256, 256))

        if self.purpose in ["hidden", "generated", "hospital_b"]:
            image = self.transform_train(image)
        else:
            image = self.transform(image)

        # 3. Determine the base ID to look for
        if self.purpose == "generated":
            # Extract the parent folder name (e.g., "0_20051020_45050_0100_PP")
            folder_name = os.path.basename(os.path.dirname(img_path))

            # Remove the prefix (e.g., "0_") to get the actual image ID
            if "_" in folder_name:
                base_id = folder_name.split("_", 1)[1]
            else:
                base_id = folder_name
        else:
            # For standard datasets, use the filename
            base_id = filename

        # 4. Retrieve Label by checking if base_id is INCLUDED in the CSV key
        label_data = {}

        for csv_key in self.labels.keys():
            if base_id in csv_key:
                label_data = self.labels[csv_key]
                break

        diagnosis = label_data.get("diagnosis", -1)

        if self.is_binary:
            # print(diagnosis, img_path)
            if int(diagnosis) >= 2:
                diagnosis = 1
            elif int(diagnosis) in [0, 1]:
                diagnosis = 0
            else:
                raise ValueError(
                    f"CRITICAL: {filename} got an invalid raw label: {diagnosis}"
                )

        return image, diagnosis, filename


# --- Testing the Logic ---
if __name__ == "__main__":
    processed_root = "/vol/biomedic3/awk24/datasets/Messidor2_256"
    csv_file = "/vol/biomedic3/awk24/datasets/Messidor2/messidor_data.csv"

    # Path to your generated samples folder
    GEN_SAMPLES_DIR = "/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/models/18_Feb_Eyepacs_DDP/simulation_GS_1.5_same_class_model_140/samples"

    print("\n--- Testing Generated Split ---")
    try:
        ds_gen = MessidorDataset(
            purpose="hospital_b",
            root_dir=processed_root,
            csv_path=csv_file,
            generated_dir=GEN_SAMPLES_DIR,
        )

        if len(ds_gen) > 0:
            # Let's inspect a few samples to ensure labels map correctly
            for i in [0, len(ds_gen) // 2, len(ds_gen) - 1]:
                img, label, fname = ds_gen[i]
                parent_folder = os.path.basename(os.path.dirname(ds_gen.image_paths[i]))
                print(
                    f"Folder: {parent_folder:<30} | File: {fname:<15} | Assigned Label: {label}"
                )

            print(f"\nTotal Generated Images Loaded: {len(ds_gen)}")

    except Exception as e:
        print(f"Error during generated split testing: {e}")
