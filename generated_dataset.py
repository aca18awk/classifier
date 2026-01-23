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
    def __init__(self, path: str, model_name: str = "efficientnet_b0", skip_early_glaucoma: bool = True):
        specs = MODEL_SPECS.get(model_name, MODEL_SPECS["default"])
        self.target_size = specs["size"]
        self.mean = specs["mean"]
        self.std = specs["std"]
        
        self.base_transform = T.Compose([T.Resize(self.target_size)])
        raw_data = ImageFolder(path, transform=self.base_transform)

        # --- Label Mapping Logic ---
        self.label_map = {} 
        self._classes = []  
        self._class_to_idx = {} 
        new_idx_counter = 0
        
        for old_idx, class_name in enumerate(raw_data.classes):
            if skip_early_glaucoma and "early" in class_name.lower():
                continue
            self.label_map[old_idx] = new_idx_counter
            self._classes.append(class_name)
            self._class_to_idx[class_name] = new_idx_counter
            new_idx_counter += 1

        # Filter indices to exclude skipped classes
        indices = [i for i, label in enumerate(raw_data.targets) if label in self.label_map]
        self.data = Subset(raw_data, indices)

        self.final_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, original_label = self.data[index] 
        label = self.label_map[original_label] # Map index 2 -> 1
        image = self.final_transform(image)
        return image, label

    @property
    def classes(self): return self._classes

    @property
    def id_to_classes(self): return {v: k for k, v in self._class_to_idx.items()}

    def len_per_class(self):
        # Accessing the underlying ImageFolder targets via the Subset indices
        subset_indices = self.data.indices
        raw_targets = [self.data.dataset.targets[i] for i in subset_indices]
        mapped_counts = Counter([self.label_map[t] for t in raw_targets])
        return {self.id_to_classes[idx]: count for idx, count in mapped_counts.items()}