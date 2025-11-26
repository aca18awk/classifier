import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from custom_model import GlaucomaClassifier
from dataset import GlaucomaHarvardDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import os

model_path = "outputs/inception_v3.tv_in1k/21_Nov_augmented_data-linear_auc/inception_v3.tv_in1k_42_best_auc.pth"
model_name = model_path.split('/')[-1].split('.pth')[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

state_dict = torch.load(model_path, map_location="cpu")
model = GlaucomaClassifier()
model.to(device)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_dataset = GlaucomaHarvardDataset("test", transform, augmentation=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

confidence = []
predicted_labels = []
actual_labels = []

for images, labels in test_loader:
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        example_out = model(images)
        probabilities = torch.nn.functional.softmax(example_out, dim=1) 
        max_probabability = [p.max().item() * 100 for p in probabilities]
        class_id = [p.argmax().item() for p in probabilities]
        confidence += max_probabability
        predicted_labels += class_id
        actual_labels += labels.tolist()


predicted_labels = np.array(predicted_labels)
actual_labels = np.array(actual_labels)

labels = [0, 1, 2]
confusion_matrix = confusion_matrix(actual_labels, predicted_labels, labels=labels)

classes_names = ['Advanced Glaucoma (0)', 'Early Glaucoma (1)', 'Healthy (2)']
report = classification_report(actual_labels, predicted_labels, target_names=classes_names, labels=labels, zero_division=0, output_dict=True)

excel_data = pd.DataFrame({
    "model path": model_path,
    "overall accuracy": report['accuracy'],
    "------": "",
    "recall (advanced glaucoma) %": report[classes_names[0]]["recall"],
    "recall (early glaucoma) %": report[classes_names[1]]["recall"],
    "recall (healthy) %": report[classes_names[2]]["recall"],
    "-------": "",
    "advanced glaucoma -> advanced glaucoma": confusion_matrix[0,0],
    "advanced glaucoma -> healthy": confusion_matrix[0,2],
    "advanced glaucoma -> early glaucoma": confusion_matrix[0,1],
    "early glaucoma -> early glaucoma": confusion_matrix[1,1],
    "early glaucoma -> healthy": confusion_matrix[1,2],
    "early glaucoma -> advanced glaucoma": confusion_matrix[1,0],
    "healthy -> healthy": confusion_matrix[2,2],
    "healthy -> advanced glaucoma": confusion_matrix[2,0],
    "healthy -> early glaucoma": confusion_matrix[2,1],
    "--------": "",
    "f1 (advanced glaucoma)": report[classes_names[0]]["f1-score"],
    "f1 (early glaucoma)": report[classes_names[1]]["f1-score"],
    "f1 (healthy)": report[classes_names[2]]["f1-score"],
    "----------": "",
    "precision (advanced glaucoma)": report[classes_names[0]]["precision"],
    "precision (early glaucoma)": report[classes_names[1]]["precision"],
    "precision (healthy)": report[classes_names[2]]["precision"],
}, index=[0])

new_column = excel_data.T.squeeze()
new_column.name = model_path

file_path = 'augmented.xlsx'

if os.path.exists(file_path):
    existing_data = pd.read_excel(file_path, index_col=0, header=None)
    combined_data = pd.concat([existing_data, new_column], axis=1)
    write_header = False 
else:
    combined_data = new_column.to_frame()
    write_header = True

print(combined_data)
combined_data.to_excel(file_path, header=write_header, index=True) 