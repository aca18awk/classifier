import os
import numbers
from typing import Literal, Dict, Optional
from collections import Counter

from torch import empty
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
# import torchvision.transforms.v2 as T
# import torchvision.transforms.functional as TF

import numpy as np

# --- Constants & Configuration ---
path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/'
DatasetSplit = Literal["test", "train", "validation"]

MODEL_SPECS = {
    "efficientnet_b0": {
        "size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225]
    },
    "inception_v3.tv_in1k": {
        "size": (299, 299),
        "mean": [0.5, 0.5, 0.5],
        "std":  [0.5, 0.5, 0.5]
    },
    # Default fallback
    "default": {
        "size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225]
    }
}

class GammaCorrectionTransform:
    """Apply Gamma Correction to the image"""
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0: # type: ignore
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)] # type: ignore
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: gamma corrected image.
        """
        gamma_factor = None if self.gamma is None else float(empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = transforms.functional.adjust_gamma(img, gamma_factor, gain=1)
        return img

class GlaucomaHarvardDataset(Dataset):
    def __init__(
        self, 
        purpose: DatasetSplit = "test", 
        model_name: str = "efficientnet_b0",
        skip_early_glaucoma: bool = True,
        # Default limits; use None or inf for no limit
        samples_per_class: Optional[Dict[str, int]] = None,
        augmentation: bool = True, 
        gen_data_path: Optional[str] = None,
        gen_counts: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            gen_data_path: Path to the generated images root folder.
            gen_counts: Dictionary specifying how many images to take per class.
                        Example: {'normal_control': 20, 'early_glaucoma': 10}
                        If a class is missing from dict, 0 images are added for that class.
        """
        # 1. Setup Model Specifics
        specs = MODEL_SPECS.get(model_name, MODEL_SPECS["default"])
        self.target_size = specs["size"]
        self.mean = specs["mean"]
        self.std = specs["std"]
        self.do_augment = augmentation

        # This transform is applied by ImageFolder immediately on load.
        # WE ONLY RESIZE HERE. We do NOT Normalize or ToTensor yet.
        self.base_transform = transforms.Compose([transforms.Resize(self.target_size)])

        # 2. Prepare Data Paths and Loading
        data_path = os.path.join(path, purpose)
        real_data = ImageFolder(data_path, transform=self.base_transform)
        
        # --- NEW LOGIC: Class Filtering & Mapping ---
        
        # 2a. Determine which classes to keep and how to map them
        # We assume folder names are roughly: 'advanced_glaucoma', 'early_glaucoma', 'normal_control'
        # Adjust the string "early" below if your folder name is different.
        
        self.label_map = {} # Maps Old_Index -> New_Index
        self._classes = []  # List of New Class Names
        self._class_to_idx = {} # New Name -> New Index
        
        new_idx_counter = 0
        
        # Iterate over the original classes found by ImageFolder (e.g., 0, 1, 2)
        for old_idx, class_name in enumerate(real_data.classes):
            # Check if we should skip this class
            if skip_early_glaucoma and "advanced" in class_name.lower():
                continue
                
            # If we keep it, record the mapping
            self.label_map[old_idx] = new_idx_counter
            self._classes.append(class_name)
            self._class_to_idx[class_name] = new_idx_counter
            new_idx_counter += 1
            
        print(f"Active Classes: {self._classes}")
        print(f"Label Mapping (Old->New): {self.label_map}")

        # 2b. Filter Real Data Indices based on Skip Logic AND Count Limits
        real_indices_to_keep = []
        
        # Track how many we have selected per class so far
        current_counts = {cls: 0 for cls in real_data.classes}

        # iterate through every image target in the real dataset (RANDOMIZED ORDER)
        all_indices = np.arange(len(real_data))
        np.random.shuffle(all_indices)

        for i in all_indices:
            original_label = real_data.targets[i]
            class_name = real_data.classes[original_label]
        
        # # iterate through every image target in the real dataset
        # for i, original_label in enumerate(real_data.targets):
        #     class_name = real_data.classes[original_label]
            
            # 1. Skip if class is not in our map (i.e., it was Early Glaucoma)
            if original_label not in self.label_map:
                continue
                
            # 2. Skip if we have reached the limit for this class
            if samples_per_class is not None and class_name in samples_per_class:
                if current_counts[class_name] >= samples_per_class[class_name]:
                    continue
            
            # If passed checks, keep index
            real_indices_to_keep.append(i)
            current_counts[class_name] += 1
            
        # Create the Subset for real data
        self.real_subset = Subset(real_data, real_indices_to_keep)
        
        # 3. Handle Generated Data (Optional)
        datasets_to_concat = [self.real_subset]
        
        if gen_data_path is not None and gen_counts is not None:
            full_gen_dataset = ImageFolder(gen_data_path, transform=self.base_transform)
            gen_indices = []
            gen_targets = np.array(full_gen_dataset.targets)
            
            for class_name, count in gen_counts.items():
                if count <= 0: continue
                
                # Check if this generated class is valid (not skipped)
                if class_name not in self._class_to_idx:
                    print(f"Skipping generated class '{class_name}' (excluded by skip_early_glaucoma or not found).")
                    continue

                # Original index in the generated folder structure
                if class_name in full_gen_dataset.class_to_idx:
                    orig_gen_idx = full_gen_dataset.class_to_idx[class_name]
                    
                    # Ensure generated dataset label mapping matches real dataset
                    # (Assuming generated folders match real folders)
                    if orig_gen_idx not in self.label_map:
                         print(f"Warning: Generated class '{class_name}' maps to index {orig_gen_idx}, which is excluded.")
                         continue

                    class_indices = np.where(gen_targets == orig_gen_idx)[0]
                    gen_indices.extend(class_indices[:count])
            
            if gen_indices:
                datasets_to_concat.append(Subset(full_gen_dataset, gen_indices))
                print(f"Added {len(gen_indices)} generated images.")

        self.data = ConcatDataset(datasets_to_concat)

        # photometric data augmentation
        self.photometric_augment = transforms.Compose([
            transforms.RandomApply(transforms=[GammaCorrectionTransform(gamma=0.3)], p=0.5),
            transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        ])

        # geometric data augmentation
        self.geometric_augment = transforms.Compose([
            transforms.RandomApply(transforms=[transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),
            transforms.RandomApply(transforms=[transforms.RandomAffine(degrees=(-10,10), scale=(0.8, 1.2))], p=0.5),
            # transforms.RandomApply(transforms=[transforms.RandomResizedCrop(scale=(0.8, 1.0), size=image_size)], p=0.5),
        ])

        # Final Normalization (Always Applied)
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # 1. Get the image and the RAW label (e.g., 2) from the dataset
        image, original_label = self.data[index] 
        
        # --- CRITICAL FIX START ---
        # We must map the raw label (2) to the new index (1)
        # If we don't do this, the model sees '2', panics, and crashes CUDA.
        if original_label in self.label_map:
            label = self.label_map[original_label]
        else:
            # Fallback (should shouldn't happen if filtering worked, but safe to have)
            print(f"Warning: Label {original_label} not in map. defaulting to 0")
            label = 0
        # --- CRITICAL FIX END ---

        if self.do_augment:
            image = self.photometric_augment(image)
            image = self.geometric_augment(image)
        
        # Always normalize at the very end
        image = self.final_transform(image)
            
        return (image, label)
    
    @property
    def classes(self):
        return self._classes
    
    @property
    def id_to_classes(self):
        return {v: k for k, v in self._class_to_idx.items()}

    def len_per_class(self):
        """
        Returns a dictionary with the count of samples per class.
        Handles ImageFolder, ConcatDataset, and Subset.
        """
        all_targets = []

        def get_targets_from_dataset(ds):
            """Helper to extract targets recursively from different dataset types."""
            if isinstance(ds, Subset):
                # If it's a Subset, we must map the indices to the parent's targets
                if hasattr(ds.dataset, 'targets'):
                    # Access parent targets using the subset's indices
                    return [ds.dataset.targets[i] for i in ds.indices]
            
            elif hasattr(ds, 'targets'):
                # Standard ImageFolder case
                return ds.targets
            
            return []

        # 1. Collect Raw Targets (e.g., 0 and 2)
        if isinstance(self.data, ConcatDataset):
            for ds in self.data.datasets:
                all_targets.extend(get_targets_from_dataset(ds))
        else:
            all_targets.extend(get_targets_from_dataset(self.data))

        # 2. Apply Mapping (Convert 0->0 and 2->1)
        # We only keep targets that exist in our label_map
        mapped_targets = [
            self.label_map[t] 
            for t in all_targets 
            if t in self.label_map
        ]

        # 3. Count Mapped Targets
        counts = Counter(mapped_targets)

        # 4. Convert Indices to Names (Now using valid keys like 0 and 1)
        class_counts = {
            self.id_to_classes[idx]: count 
            for idx, count in counts.items()
        }
        
        # Ensure all classes are present (even if count is 0)
        for class_name in self.classes:
            if class_name not in class_counts:
                class_counts[class_name] = 0

        return class_counts


# Experiment 1: Specific mix

# gen_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/2025/Nov_3_conditional_model/with_classifer_10'
# counts_exp_1 = {
#     "early_glaucoma": 10,
#     "normal_control": 20,
#     "advanced_glaucoma": 15
# }

limits = {
    "early_glaucoma": 5,
    "normal_control": 5,
    # "early_glaucoma": 50 # This will be ignored automatically due to skip flag
}

dataset_exp1 = GlaucomaHarvardDataset(
    purpose="train", 
    augmentation=False,
    skip_early_glaucoma=True,  # Enable the filter
    samples_per_class=limits   # Apply the limits
)

print(f"Total dataset size: {len(dataset_exp1)}")
print("Counts per class:", dataset_exp1.len_per_class())
# Check one item to ensure label is an integer (0 or 1)
img, lbl = dataset_exp1[0]
print(f"Sample Label: {lbl} (Class: {dataset_exp1.id_to_classes[lbl]})")