from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2 as T
from collections import Counter


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

class GeneratedGlaucomaDataset(Dataset):
    def __init__(
        self, 
        path: str,
        model_name: str = "efficientnet_b0",
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

        
        data_path = path
        
        # 1. Load Real Data
        real_data = ImageFolder(data_path, transform=self.base_transform)
        self._classes = real_data.classes
        self._class_to_idx = real_data.class_to_idx

        self.data = real_data

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

# dataset_exp1 = GlaucomaHarvardDataset(
#     purpose="train", 
#     augmentation=False,
#     gen_data_path=gen_path,
#     gen_counts=counts_exp_1
# )

# print(f"Total dataset size: {len(dataset_exp1)}")
# counts = dataset_exp1.len_per_class()
# print("Counts per class:", counts)
# print(dataset_exp1[1])
