import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchmetrics.functional import auroc


import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from dataset import GlaucomaHarvardDataset
from custom_model import GlaucomaClassifier
import torch.optim.lr_scheduler as lr_scheduler
import wandb

# CONSTANTS
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--num_epoch", default=5)
parser.add_argument("--model_name", default="efficientnet_b0")
args = parser.parse_args()
config = vars(args)

print("Arguments: ", config)

learning_rate = float(config["learning_rate"])
num_epoch = int(config["num_epoch"])
model_name = config["model_name"]

output_dir = os.path.join("outputs", model_name, "25_Nov_gen_data_2")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

AI_DATA_PATH = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/25_Nov/lunar-shadow-27_GS_1.0'
AMOUNT_ADDED_PERCENT = 0.003
AMOUNT_ADDED = int(AMOUNT_ADDED_PERCENT * 1079)
count_early = int(0.2 * AMOUNT_ADDED)
count_normal = int(0.3 * AMOUNT_ADDED)
count_advanced = AMOUNT_ADDED - (count_early + count_normal)
counts_exp_1 = {
    "early_glaucoma": count_early,
    "normal_control": count_normal,
    "advanced_glaucoma": count_advanced
}

logger =  wandb.init(
                project="classifier_train_fixed",
                # name="data augm (17 Nov) - inception_v3 CosineAnnealingLR",
                config={
                    "epoch": num_epoch,
                    "model_name": model_name,
                    "learning_rate": learning_rate,
                    # "augmented_data_path": AI_DATA_PATH,
                    # "ratio of data added": AMOUNT_ADDED_PERCENT,
                    # "exact numbers added": counts_exp_1
                },
                resume="allow",
            )

run_name = logger.name 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


transform = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform1 = transforms.Compose([
    transforms.Resize([299, 299]),
    transforms.ToTensor(),
    transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
    )
])

# train_dataset = GlaucomaHarvardDataset("train", transform1, gen_counts=counts_exp_1, gen_data_path=AI_DATA_PATH)
train_dataset = GlaucomaHarvardDataset("train", transform1, augmentation=False)
test_dataset = GlaucomaHarvardDataset("test", transform1, augmentation=False)
val_dataset = GlaucomaHarvardDataset("validation", transform1, augmentation=False)

print(train_dataset.id_to_classes)

# splits the data into batches of size 32. So each it returns the array of arrays of 32 images each.
# it returns batches of images and batches of labels
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    print(images.shape, labels.shape)
    break


model = GlaucomaClassifier(model_name=model_name)
model.to(device)

example_out = model(images)
print(example_out.shape)

print("Running Shape Check...")
dummy_input = torch.randn(2, 3, 299, 299).to(device) # Batch of 2 images
dummy_output = model(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {dummy_output.shape}")


# training script
print("training phase: ")

criterion = nn.CrossEntropyLoss() # loss function
# optimizer = optim.Adam(model.parameters(), lr=learning_rate) # learning rate is constant. Can add learning rate scheduler later

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
print("lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)")
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
# print("scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)")

train_losses, val_losses = [], []

best_val_loss = float('inf') # <--- FIXED
best_model = None
best_epoch = 0
best_model_auc = 0
best_epoch_auc = 0
best_val_auc = 0
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    train_preds = []
    train_trgts = []
    for images,labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        train_preds.append(outputs.cpu()) 
        train_trgts.append(labels.cpu())




    train_preds = torch.cat(train_preds, dim=0)
    train_trgts = torch.cat(train_trgts, dim=0) 
    auc = auroc(train_preds, train_trgts, num_classes=3, average='macro', task='multiclass')

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr)) 

    train_loss = running_loss / train_dataset.__len__()
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    val_preds = []
    val_trgts = []

    with torch.no_grad():
        for images,labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            val_preds.append(outputs.cpu()) 
            val_trgts.append(labels.cpu())
    
        val_loss = running_loss / val_dataset.__len__()
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict() # <--- This ensures best_model is not None
            best_epoch = epoch

        val_preds = torch.cat(val_preds, dim=0)
        val_trgts = torch.cat(val_trgts, dim=0) 
        auc_val = auroc(val_preds, val_trgts, num_classes=3, average='macro', task='multiclass').item()
        if auc_val > best_val_auc:
            best_val_auc = auc_val
            best_model_auc = model.state_dict()
            best_epoch_auc= epoch

    logger.log({"epoch":epoch, "val_loss":val_loss, "train_loss":train_loss, "lr":after_lr, "auc": auc, "auc_val":auc_val})
    

    print(f"Epoch {epoch+1}/{num_epoch} - Train loss: {train_loss}, Validation loss: {val_loss}")

torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_data_{AMOUNT_ADDED_PERCENT}_last.pth"))
torch.save(best_model, os.path.join(output_dir, f"{run_name}_{best_epoch}_data_{AMOUNT_ADDED_PERCENT}_best_.pth"))
torch.save(best_model_auc, os.path.join(output_dir, f"{run_name}_{best_epoch_auc}_data_{AMOUNT_ADDED_PERCENT}_best_auc.pth"))


plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.savefig(f"{output_dir}/loss.png")
print("training completed")
