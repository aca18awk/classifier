import os

from typing import Literal
from enum import Enum

from torch import unsqueeze, from_numpy, empty, Tensor
from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from skimage.io import imread
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
import numpy as np
import numbers
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from typing import Literal, Dict, Optional
from collections import Counter

DatasetSplit = Literal["test", "train", "validation"]

class DatasetClass(Enum):
    normal_control = 0
    early_glaucoma = 1
    advanced_glaucoma = 2

path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/'

AI_gen = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/2025/Nov_3_conditional_model/with_classifer_10'

# torch randaugment - do standard augmentations
#  colour, spatial rotations, not crazy rotations
# track macro AUC in torch metrics 
# early stopping based on AUC - save the best model based on that
# H-VAE, different diffusion models
# (common diffusion model, Flow matching model)

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
        transform = None, 
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
        data_path = os.path.join(path, purpose)
        
        # 1. Load Real Data
        real_data = ImageFolder(data_path, transform=transform)
        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx
        self.do_augment = augmentation
        
        datasets_to_concat = [real_data]

        # 2. Load and Filter Generated Data
        if gen_data_path is None or gen_counts is None:
            self.data = real_data
        else:
            full_gen_dataset = ImageFolder(gen_data_path, transform=transform)
            
            indices_to_include = []
            
            # ImageFolder.targets contains the list of class indices for every image
            # We convert targets to a numpy array for easier indexing
            all_targets = np.array(full_gen_dataset.targets)
            
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
                # Create a subset with only the selected indices
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

        self.processing_normalize = T.Compose([
            T.ToTensor(),
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))but
        ])
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if not self.do_augment:
            return self.data[index]
        else:
            image, label = self.data[index]
            
            # Apply Augmentations
            # image = self.photometric_augment(image) # Uncomment if needed
            image = self.geometric_augment(image)
            
            if not isinstance(image, Tensor):
                 image = T.ToTensor()(image)
                 
            image = self.processing_normalize(image)
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

# transform = transforms.Compose([
#     transforms.Resize([128, 128]),
#     transforms.ToTensor()
# ])
# dataset1 = GlaucomaHarvardDataset("train", transform=transform)
# # print(dataset1.__len__())
# # dataset1.classes
# print(dataset1.id_to_classes)
# counts = dataset1.len_per_class()
# print("Counts per class:", counts)

# gen_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/2025/Nov_3_conditional_model/with_classifer_10'

# # Experiment 1: Specific mix
# counts_exp_1 = {
#     "early_glaucoma": 10,
#     "normal_control": 20,
#     "advanced_glaucoma": 15
# }

# dataset_exp1 = GlaucomaHarvardDataset(
#     purpose="train", 
#     augmentation=True,
#     gen_data_path=gen_path,
#     gen_counts=counts_exp_1
# )

# print(f"Total dataset size: {len(dataset_exp1)}")
# counts = dataset_exp1.len_per_class()
# print("Counts per class:", counts)


# image, label = dataset1.__getitem__(0)
# print(image)
# # Save the image to a file
# output_path = os.path.join("outputs", "output_image.png")
# save_image(image, output_path)
# print(f"Image saved to {output_path}")



transform_pipe = transforms.Compose([
    transforms.ToPILImage(), # Convert np array to PILImage
    transforms.Resize(
        size=(224, 224)
    ),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # ),
])

# MORE MANUAL DEFINITION
class GlaucomaHarvardDatasetOld(Dataset):
    """
    Dataset class needs to have those 3 methods overwritten
    init - what to do when dataset is created
    len - model needs to know how big is the dataset
    getitem - to get specific item by using an id
    """
    
    def __init__(self, purpose:DatasetSplit = "test", transform=transform_pipe):

        dataPath = os.path.join(path,purpose)

        files = []
        labels = []
        for datasetClass in DatasetClass: 
            folderPath = os.path.join(dataPath, datasetClass.name)

            newfiles = [os.path.join(folderPath, filename) for filename in os.listdir(folderPath) if filename.endswith(".png")]
            files += newfiles
            labels += [datasetClass.value] * len(newfiles)
        
        self.images = files
        self.labels = labels
            
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = imread(img_path)
        
        if self.transform:
            img = self.transform(img)
            img = unsqueeze(img, 0)
        
        sample = {
            "image": img,
            "label": self.labels[idx],
            "id": os.path.basename(self.images[idx]).replace(".png", "")
        }

        return sample
    
    def __len__(self):
        return len(self.images)



# dataset = GlaucomaHarvardDataset("test")
# dataset.__getitem__(1)
# dataset.__len__()