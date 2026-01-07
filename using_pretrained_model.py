from torchvision.models import resnet50, ResNet50_Weights
import os

from typing import Literal
from enum import Enum

from torch import unsqueeze
from torch.utils.data import Dataset
from skimage.io import imread
from torchvision import transforms


DatasetSplit = Literal["test", "train", "validation"]

class DatasetClass(Enum):
    normal_control = 0
    early_glaucoma = 1
    advanced_glaucoma = 2

path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/'

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


weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
train_data = GlaucomaHarvardDatasetOld("train")
example_image = train_data.__getitem__(0)['image']

model = resnet50(weights=weights)
model.eval()

predictions = model(example_image).squeeze(0).softmax(0)
class_id = predictions.argmax().item()

print(f'Class ID: {class_id}, category: {weights.meta["categories"][class_id]}')
print(f' with confidence: {predictions[class_id].item()}')
