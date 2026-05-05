import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from collections import defaultdict
import copy
import os
import numpy as np
from timeit import default_timer
from tqdm import tqdm

# 0. Device Setting - Check if GPU is available
# ---------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Training on : ", device)

# PATHS
# ---------------------------------------------
root_dir = ""
data_dir = os.path.join(root_dir, "Data")
img_dir = os.path.join(data_dir, "ChestXRay/image")
mask_dir = os.path.join(data_dir, "ChestXRay/mask")
metadata_path = os.path.join(data_dir, "MetaData.csv")
path_model = os.path.join(root_dir, "Model/TBC_Classifier.pt")
# ---------------------------------------------

# Datasets - Train/Validation
# ---------------------------------------------
from Dataset.Classification_Dataset import TBCDataset
img_size = 256
train_set = TBCDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    metadata_path=metadata_path,
    size=(img_size, img_size),
    split="train",
    train_ratio=0.7
)
val_set = TBCDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    metadata_path=metadata_path,
    size=(img_size, img_size),
    split="val",
    train_ratio=0.7
)

# Dataloaders
# ---------------------------------------------
batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model
# ---------------------------------------------
from TBC_Classifier import TBC_Classifier, count_params, get_lr
enc_ch = [64,64,32]
#enc_ch = [256,128,64]
#enc_ch = [64, 128, 256]
model = TBC_Classifier(
    in_channels=1,
    encoder_depth=3,
    encoder_channels=enc_ch,
).to(device)
print(model)
print(f"Model Parameters : {count_params(model)}")

# Hyperparameters
Loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3,
                              weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.8,
                                                       patience=2,
                                                       cooldown=1,
                                                       min_lr=0)

# initialize variables
# -------------------------------------------
history = defaultdict(list)
min_loss     = np.inf
patience     = 0
max_patience = 25
best_model = copy.deepcopy(model.state_dict())

# Training loop
# -------------------------------------------
epoch = 0
max_epoch = 100
start = default_timer()

print("Start Training ...")
while True:
    model.train()
    t1 = default_timer()

    train_loss = 0.0
    conf = torch.zeros(2, 2, dtype=torch.long)
    total_samples = 0

    # Train loader iteration
    for _, data in enumerate(tqdm(train_loader)):
        x, y = data
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(x)  # shape : [B,C,S,S]

        # Compute binary cross-entropy loss
        y = y.float().unsqueeze(1)  # (B,) -> (B,1)

        bce = Loss(out, y)
        bce.backward()  # Backpropagation

        optimizer.step()
        train_loss += bce.item()

        # Update confusion matrix as running inspection
        preds = (torch.sigmoid(out) > 0.5).long().squeeze(1).cpu()
        targets = y.long().squeeze(1).cpu()

        conf[0, 0] += ((preds == 0) & (targets == 0)).sum()
        conf[0, 1] += ((preds == 1) & (targets == 0)).sum()
        conf[1, 0] += ((preds == 0) & (targets == 1)).sum()
        conf[1, 1] += ((preds == 1) & (targets == 1)).sum()

        total_samples += batch_size

    t_accuracy = (conf.diag().sum() / conf.sum()).item()
    train_loss /= total_samples

    # Validation
    model.eval()
    val_loss = 0.0
    conf = torch.zeros(2, 2, dtype=torch.long)
    total_samples = 0

    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader)):
            x, y = data
            batch_size = x.size(0)
            x, y = x.to(device), y.to(device)

            out = model(x)
            y = y.float().unsqueeze(1)  # (B,) -> (B,1)
            val_loss += Loss(out, y).item()

            # Update confusion matrix as running inspection
            preds = (torch.sigmoid(out) > 0.5).long().squeeze(1).cpu()
            targets = y.long().squeeze(1).cpu()

            conf[0, 0] += ((preds == 0) & (targets == 0)).sum()
            conf[0, 1] += ((preds == 1) & (targets == 0)).sum()
            conf[1, 0] += ((preds == 0) & (targets == 1)).sum()
            conf[1, 1] += ((preds == 1) & (targets == 1)).sum()

            total_samples += batch_size

    accuracy = (conf.diag().sum() / conf.sum()).item()
    val_loss /= total_samples

    # --------------------------------
    history["lrs"].append(get_lr(optimizer))
    history["train_losses"].append(train_loss)
    history["val_losses"].append(val_loss)
    history["train_acc"].append(t_accuracy)
    history["val_acc"].append(accuracy)

    t2 = default_timer()
    print(f"Epoch : {epoch}, "
          f"| Train Loss : {np.round(train_loss, 5)}, "
          f"| Train Accuracy : {np.round(t_accuracy, 3)}, "
          f"| LR : {get_lr(optimizer)}")
    print(f"Val Loss : {np.round(val_loss, 5)}, "
          f"| Val Accuracy : {np.round(accuracy, 3)}, "
          f"| Time : {np.round(t2 - t1, 2)}")
    print("Confusion Matrix :")
    print(conf.cpu().numpy())

    cur_val_loss = val_loss
    if cur_val_loss < min_loss:
        print('Loss Decreasing.. {:.5f} >> {:.5f} '.format(min_loss, cur_val_loss))
        min_loss = cur_val_loss
        best_model = copy.deepcopy(model.state_dict())
        patience = 0

        # End if the model reach perfect classification
        if conf[0, 1] + conf[1, 0] == 0:
            break

    else:
        patience += 1
        print(f'Loss Not Decrease for {patience} time')

        # Ending Condition
        if patience == max_patience:
            print('Loss not decrease for {} times, Stop Training ...'.format(patience))
            break

    if epoch >= max_epoch:
        print("Maximum number of epochs reached, stop training ...")
        break

    epoch += 1
    scheduler.step(cur_val_loss)

fit_time = default_timer()
print("Total time = ", np.round((fit_time - start) / 60, 2), "m")
torch.save(best_model, path_model)

def plot_results(history, model_name):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))
    fig.suptitle(model_name, fontsize=16)

    # --- Loss ---
    axes[0].plot(history["train_losses"], label="Train")
    axes[0].plot(history["val_losses"], label="Val")
    axes[0].set_yscale("log")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    # --- Accuracy ---
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid()

    # --- Learning Rate Scheduling ---
    axes[2].plot(history["lrs"])
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Results", bbox_inches="tight")
    plt.show()

plot_results(history, "TBC_Classifier")
