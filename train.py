import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc
import copy


import os
import argparse

from dataset import GlaucomaHarvardDataset
from custom_model import GlaucomaClassifier
import torch.optim.lr_scheduler as lr_scheduler
import wandb

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


# CONSTANTS
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--num_epoch", default=5)
parser.add_argument("--model_name", default="efficientnet_b0")
parser.add_argument("--augment_with_ai", type=str2bool, default=False)
parser.add_argument("--augment", type=str2bool, default=True)
args = parser.parse_args()
config = vars(args)

print("Arguments: ", config)

learning_rate = float(config["learning_rate"])
num_epoch = int(config["num_epoch"])
model_name = config["model_name"]
augment_with_ai = config["augment_with_ai"]
augment = config["augment"]

if augment_with_ai:
    ai_data_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/25_Nov/lunar-shadow-27_GS_5.0'
    AMOUNT_ADDED_PERCENT = 5
    AMOUNT_ADDED = int(AMOUNT_ADDED_PERCENT * 1079)
    count_early = int(0.2 * AMOUNT_ADDED)
    count_normal = int(0.3 * AMOUNT_ADDED)
    count_advanced = AMOUNT_ADDED - (count_early + count_normal)
    counts= {
        "early_glaucoma": count_early,
        "normal_control": count_normal,
        "advanced_glaucoma": count_advanced
    }
else:
    AMOUNT_ADDED_PERCENT = 0
    ai_data_path = None
    counts = None

output_dir = os.path.join("outputs2", model_name, f"26_Nov_gen_data_{AMOUNT_ADDED_PERCENT}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for i in range(5):
    logger =  wandb.init(
                    project="classifier_train_experiments",
                    config={
                        "epoch": num_epoch,
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "data augmentation": augment,
                        "AI data added": augment_with_ai,
                        "augmented_data_path": ai_data_path,
                        "ratio of data added": AMOUNT_ADDED_PERCENT,
                        "exact numbers added": counts
                    },
                    resume="allow",
                )

    run_name = logger.name 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)


    train_dataset = GlaucomaHarvardDataset("train", model_name=model_name, augmentation=augment, gen_counts=counts, gen_data_path=ai_data_path)
    test_dataset = GlaucomaHarvardDataset("test", model_name=model_name, augmentation=False)
    val_dataset = GlaucomaHarvardDataset("validation", model_name=model_name, augmentation=False)

    print(train_dataset.id_to_classes)

    # splits the data into batches of size 32. So each it returns the array of arrays of 32 images each.
    # it returns batches of images and batches of labels
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = GlaucomaClassifier(model_name=model_name)
    model.to(device)

    print("training phase: ")
    criterion = nn.CrossEntropyLoss() # loss function

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate) # learning rate is constant. Can add learning rate scheduler later
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print(f"lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters={num_epoch})")
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epoch)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # print("scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)")

    best_val_loss = float('inf') 
    best_model = None
    best_epoch = 0
    best_model_auc = None
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

            train_preds.append(outputs.detach().cpu()) 
            train_trgts.append(labels.detach().cpu())


        train_preds = torch.cat(train_preds, dim=0)
        train_trgts = torch.cat(train_trgts, dim=0) 
        train_auc = auroc(train_preds, train_trgts, num_classes=3, average='macro', task='multiclass')
        train_loss = running_loss / len(train_dataset)
        

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        running_loss = 0.0
        val_preds = []
        val_trgts = []

        with torch.no_grad():
            for images,labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                val_preds.append(outputs.cpu()) 
                val_trgts.append(labels.cpu())
        
            val_loss = running_loss / len(val_dataset)

            val_preds = torch.cat(val_preds, dim=0)
            val_trgts = torch.cat(val_trgts, dim=0) 
            val_auc = auroc(val_preds, val_trgts, num_classes=3, average='macro', task='multiclass').item()


            print(f"Epoch {epoch+1}/{args.num_epoch} | LR: {after_lr:.5f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            logger.log({
                "epoch": epoch, 
                "val_loss": val_loss, 
                "train_loss": train_loss, 
                "lr": after_lr, 
                "train_auc": train_auc, 
                "val_auc": val_auc
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch


            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_auc = copy.deepcopy(model.state_dict())
                best_epoch_auc= epoch
        
    torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_data_{AMOUNT_ADDED_PERCENT}_last.pth"))
    if best_model is not None:
        torch.save(best_model, os.path.join(output_dir, f"{run_name}_{best_epoch}_data_{AMOUNT_ADDED_PERCENT}_best_loss.pth"))
    if best_model_auc is not None:
        torch.save(best_model_auc, os.path.join(output_dir, f"{run_name}_{best_epoch_auc}_data_{AMOUNT_ADDED_PERCENT}_best_auc.pth"))