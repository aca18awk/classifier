import os

from typing import Literal

from torch import empty
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import numpy as np
import numbers
from typing import Literal, Dict, Optional
from collections import Counter


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
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img

class GlaucomaHarvardDataset(Dataset):
    def __init__(
        self, 
        purpose: DatasetSplit = "test", 
        model_name: str = "efficientnet_b0",
        augmentation = True, 
        gen_data_path = None,
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
        
        # This transform is applied by ImageFolder immediately on load.
        # WE ONLY RESIZE HERE. We do NOT Normalize or ToTensor yet.
        self.base_transform = T.Compose([
            T.Resize(self.target_size),
        ])

        
        data_path = os.path.join(path, purpose)
        
        # 1. Load Real Data
        real_data = ImageFolder(data_path, transform=self.base_transform)
        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx
        self.do_augment = augmentation

        if gen_data_path is None or gen_counts is None:
            self.data = real_data

        # 2. Load and Filter Generated Data
        else:
            datasets_to_concat = [real_data]
            full_gen_dataset = ImageFolder(gen_data_path, transform=self.base_transform)
            
            indices_to_include = []
            
            # ImageFolder.targets contains the list of class indices for every image
            # We convert targets to a numpy array for easier indexing
            all_targets = np.array(full_gen_dataset.targets)
            print(all_targets)
            
            for class_name, count in gen_counts.items():
                if count <= 0:
                    continue
                
                # Get the integer label for the class name
                if class_name in full_gen_dataset.class_to_idx:
                    class_idx = full_gen_dataset.class_to_idx[class_name]
                    
                    # Find all indices in the dataset that match this class
                    class_indices = np.where(all_targets == class_idx)[0]
                    
                    # Select the top N indices (or all if count > available)
                    selected_indices = class_indices[:count]
                    indices_to_include.extend(selected_indices)
                else:
                    print(f"Warning: Class '{class_name}' found in gen_counts but not in generated dataset folder.")

            if len(indices_to_include) > 0:
                gen_subset = Subset(full_gen_dataset, indices_to_include)
                datasets_to_concat.append(gen_subset)
                print(f"Added {len(indices_to_include)} generated images to the dataset.")
            else:
                print("No generated images added (gen_counts resulted in 0 samples).")

            self.data = ConcatDataset(datasets_to_concat)

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            T.RandomApply(transforms=[GammaCorrectionTransform(gamma=0.3)], p=0.5),
            T.RandomApply(transforms=[T.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=0.0, p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
        ])

        # geometric data augmentation
        self.geometric_augment = T.Compose([
            T.RandomApply(transforms=[T.RandomPerspective(distortion_scale=0.2)], p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=(-10,10), scale=(0.8, 1.2))], p=0.5),
            # T.RandomApply(transforms=[T.RandomResizedCrop(scale=(0.8, 1.0), size=image_size)], p=0.5),
        ])

        # Final Normalization (Always Applied)
        self.final_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # image is PIL because ImageFolder used only Resize
        image, label = self.data[index] 
        
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
                    # Note: ImageFolder targets are usually a list
                    return [ds.dataset.targets[i] for i in ds.indices]
            
            elif hasattr(ds, 'targets'):
                # Standard ImageFolder case
                return ds.targets
            
            return []

        # Main logic to aggregate targets
        if isinstance(self.data, ConcatDataset):
            for ds in self.data.datasets:
                all_targets.extend(get_targets_from_dataset(ds))
        else:
            all_targets.extend(get_targets_from_dataset(self.data))

        # Count occurrences
        counts = Counter(all_targets)

        # Map class indices to class names
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

dataset_exp1 = GlaucomaHarvardDataset(
    purpose="train", 
    augmentation=False,
    # gen_data_path=gen_path,
    # gen_counts=counts_exp_1
)

print(f"Total dataset size: {len(dataset_exp1)}")
counts = dataset_exp1.len_per_class()
print("Counts per class:", counts)
print(dataset_exp1[1])
