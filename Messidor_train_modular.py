import argparse
import copy
import os
import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics.functional import auroc

import wandb
from Messidor_class_generated import MessidorDataset


class Classifier(nn.Module):
    def __init__(self, model_name="resnet18", num_classes=3, in_chans=3):
        super(Classifier, self).__init__()
        self.base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_chans,
        )

    def forward(self, x):
        return self.base_model(x)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# CONSTANTS
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="resnet18")
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--num_epoch", default=100)
parser.add_argument("--is_binary", action="store_true")
parser.add_argument("--split", default="augmented")
parser.add_argument(
    "--generated_dir",
    default=None,
)

args = parser.parse_args()
config = vars(args)

print("Arguments: ", config)

learning_rate = float(config["learning_rate"])
num_epoch = int(config["num_epoch"])
model_name = config["model_name"]
is_binary = bool(config["is_binary"])
split = config["split"]
generated_dir = config["generated_dir"]

if split not in ["augmented", "hospital_b", "hidden"]:
    raise ValueError(f"CRITICAL: split got an invalid value: {split}")

if split == "augmented" and generated_dir is None:
    raise ValueError("Specify generated_dir")

num_classes = 2 if is_binary else 5

GEN_SAMPLES_DIR = None
if generated_dir is not None:
    GEN_SAMPLES_DIR = os.path.join(
        "/vol/biomedic3/awk24/code/conditional-flow-matching/examples/images/models/",
        generated_dir,
    )

IMG_SIZE = 128

SEEDS = [42, 44, 45, 50, 7]
project = "classifier_Messidor_binary" if is_binary else "classifier_Messidor"

# create dirs
main_dir = "Messidor_binary" if is_binary else "Messidor_5_class"
output_dir = os.path.join(main_dir, model_name, f"24_Feb_{split}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for run_seed in SEEDS:
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)

    logger = wandb.init(
        project=project,
        config={
            "epoch": num_epoch,
            "model_name": model_name,
            "learning_rate": learning_rate,
            "run_seed": run_seed,
            "split": split,
            "feature": generated_dir,
        },
        resume="allow",
    )

    run_name = logger.name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    if split == "hidden":
        train_dataset = MessidorDataset(
            "hidden", img_size=IMG_SIZE, is_binary=is_binary
        )
    elif split == "hospital_b":
        train_dataset = MessidorDataset(
            "hospital_b", img_size=IMG_SIZE, is_binary=is_binary
        )
    else:
        dataset_gen = MessidorDataset(
            "generated",
            img_size=IMG_SIZE,
            generated_dir=GEN_SAMPLES_DIR,
            is_binary=is_binary,
        )
        dataset_real = MessidorDataset(
            "hospital_b", img_size=IMG_SIZE, is_binary=is_binary
        )
        train_dataset = ConcatDataset([dataset_gen, dataset_real])
        print(
            f"Combined Training Set Size: {len(train_dataset)} ({len(dataset_gen)} gen + {len(dataset_real)} real)"
        )

    val_dataset = MessidorDataset("validation", img_size=IMG_SIZE, is_binary=is_binary)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Classifier(model_name=model_name, num_classes=num_classes, in_chans=3)
    model.to(device)

    print("training phase: ")
    if split == "hidden" and not is_binary:
        class_weights = torch.tensor([0.0639, 0.1335, 0.1093, 0.2656, 0.4277]).to(
            device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate) # learning rate is constant. Can add learning rate scheduler later
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    # scheduler = lr_scheduler.LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.5, total_iters=num_epoch
    # )
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # print("scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)")

    best_val_loss = float("inf")
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

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)

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

        train_auc = auroc(
            train_preds,
            train_trgts,
            average="macro",
            num_classes=num_classes,
            task="multiclass",
        )
        train_loss = running_loss / len(train_dataset)

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        running_loss = 0.0
        val_preds = []
        val_trgts = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.long)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                val_preds.append(outputs.cpu())
                val_trgts.append(labels.cpu())

            val_loss = running_loss / len(val_dataset)

            val_preds = torch.cat(val_preds, dim=0)
            val_trgts = torch.cat(val_trgts, dim=0)

            val_auc = auroc(
                val_preds,
                val_trgts,
                average="macro",
                num_classes=num_classes,
                task="multiclass",
            ).item()

            print(
                f"Epoch {epoch + 1}/{args.num_epoch} | LR: {after_lr:.5f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

            logger.log(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "lr": after_lr,
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_auc = copy.deepcopy(model.state_dict())
                best_epoch_auc = epoch

    if best_model_auc is not None:
        torch.save(
            best_model_auc,
            os.path.join(
                output_dir,
                f"{run_name}_{best_epoch_auc}_best_auc.pth",
            ),
        )

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"{run_name}_last.pth"),
    )

    wandb.finish()
