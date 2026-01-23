import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc
import copy
import random
import numpy as np


import os
import argparse


from dataset import GlaucomaHarvardDataset
from custom_model import GlaucomaClassifier
import torch.optim.lr_scheduler as lr_scheduler
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def inspect_dataset_labels(dataset, name="Dataset", limit=None):
    """
    Iterates through the dataset and prints labels to verify they are correct.
    Args:
        dataset: The dataset object
        name: Name for printing
        limit: If set (e.g., 20), only prints the first 20 items. Set to None to print all.
    """
    print(f"\n--- Inspecting Labels for {name} ({len(dataset)} items) ---")
    print(f"{'Idx':<6} | {'Label':<6} | {'Class Name'}")
    print("-" * 35)
    
    # Collect all unique labels found to verify at the end
    found_labels = set()

    max_range = len(dataset) if limit is None else min(len(dataset), limit)

    for i in range(max_range):
        # dataset[i] returns (image, label). We only care about label [1]
        # Note: This calls __getitem__, so it applies your mapping logic (0->0, 2->1)
        _, label = dataset[i]
        
        class_name = dataset.id_to_classes.get(label, "UNKNOWN")
        found_labels.add(label)
        
        print(f"{i:<6} | {label:<6} | {class_name}")

    print("-" * 35)
    print(f"Unique labels found in {name}: {found_labels}")
    if limit is not None and len(dataset) > limit:
        print(f"... (showing first {limit} of {len(dataset)} images)")
    print("\n")

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
parser.add_argument("--skip_early_glaucoma", type=str2bool, default=True)
parser.add_argument("--num_per_class", default=None)
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
skip_early_glaucoma = config["skip_early_glaucoma"]
num_per_class = config["num_per_class"]
num_classes = 2 if skip_early_glaucoma else 3


flow_matching_path = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/18_Dec/fluent-morning-29_GS_20.0'
flow_matching_path_2 = '/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/18_Dec/fluent-morning-29_GS_10.0'
diffusion_path = "/vol/biomedic3/awk24/datasets/Glaucoma_fundus/generated/Jan_15_diff_conditional_model_90000_DDPM_2"

if augment_with_ai:
    ai_data_path = diffusion_path
    AMOUNT_ADDED_PERCENT = 35
    # AMOUNT_ADDED = int(0.01 * AMOUNT_ADDED_PERCENT * 1079)
    # count_early = int(0.2 * AMOUNT_ADDED)
    # count_normal = int(0.3 * AMOUNT_ADDED)
    # count_advanced = AMOUNT_ADDED - (count_early + count_normal)
    counts= {
        "early_glaucoma": AMOUNT_ADDED_PERCENT,
        "normal_control": AMOUNT_ADDED_PERCENT,
        "advanced_glaucoma": 0
    }
else:
    AMOUNT_ADDED_PERCENT = 0
    ai_data_path = None
    counts = None


experiment_group_name = f"21_Jan_real_randomised_data_all_classes_{AMOUNT_ADDED_PERCENT}"

for num_per_class in [1,5, 10, 15, 20, 30, 40, 50, 100, 200]:
    samples_per_class = None if num_per_class==None else {
    "early_glaucoma": int(num_per_class),
    "normal_control": int(num_per_class),
    "advanced_glaucoma": int(num_per_class)
    }

    # output_dir = os.path.join("outputs2", model_name, f"7_Jan_gen_data_{AMOUNT_ADDED_PERCENT}")
    output_dir = os.path.join("outputs_all_classes", model_name, f"21_Jan/with_{num_per_class}_real")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(5):
        # Set a unique seed for this specific run (e.g., 0, 1, 2, 3, 4)
        run_seed = i 
        
        # Apply seed to all random generators
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)
        logger =  wandb.init(
                        project="classifier_limited_data",
                        group=experiment_group_name,
                        config={
                            "epoch": num_epoch,
                            "model_name": model_name,
                            "learning_rate": learning_rate,
                            "data augmentation": augment,
                            "AI data added": augment_with_ai,
                            "augmented_data_path": ai_data_path,
                            # "ratio of data added": AMOUNT_ADDED_PERCENT,
                            "exact numbers added": counts,
                            "skip_early_glaucoma": skip_early_glaucoma,
                            "samples_per_class": samples_per_class,
                            "run_seed": run_seed,
                        },
                        resume="allow",
                    )

        run_name = logger.name 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device: ", device)


        train_dataset = GlaucomaHarvardDataset("train", model_name=model_name, augmentation=augment, gen_counts=counts, gen_data_path=ai_data_path, skip_early_glaucoma=skip_early_glaucoma, samples_per_class=samples_per_class)
        val_dataset = GlaucomaHarvardDataset("validation", model_name=model_name, augmentation=False, skip_early_glaucoma=skip_early_glaucoma)

        print(train_dataset.id_to_classes)
        inspect_dataset_labels(train_dataset, name="Train Dataset", limit=None)
        inspect_dataset_labels(val_dataset, name="Val Dataset", limit=None)

        # splits the data into batches of size 32. So each it returns the array of arrays of 32 images each.
        # it returns batches of images and batches of labels
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = GlaucomaClassifier(model_name=model_name, num_classes=num_classes)
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
            train_auc = auroc(train_preds, train_trgts, num_classes=num_classes, average='macro', task='multiclass')
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
                val_auc = auroc(val_preds, val_trgts, num_classes=num_classes, average='macro', task='multiclass').item()


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
            
        # torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_data_{AMOUNT_ADDED_PERCENT}_last.pth"))
        # if best_model is not None:
        #     torch.save(best_model, os.path.join(output_dir, f"{run_name}_{best_epoch}_data_{AMOUNT_ADDED_PERCENT}_best_loss.pth"))
        # if best_model_auc is not None:
        #     torch.save(best_model_auc, os.path.join(output_dir, f"{run_name}_{best_epoch_auc}_data_{AMOUNT_ADDED_PERCENT}_best_auc.pth"))
        # if best_model is not None:
        #     torch.save(best_model, os.path.join(output_dir, f"{run_name}_{best_epoch}_data_{AMOUNT_ADDED_PERCENT}_best_loss.pth"))
        if best_model_auc is not None:
            torch.save(best_model_auc, os.path.join(output_dir, f"{run_name}_{best_epoch_auc}_data_{AMOUNT_ADDED_PERCENT}_best_auc.pth"))
        else: 
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}_data_{AMOUNT_ADDED_PERCENT}_last.pth"))
        
        wandb.finish()