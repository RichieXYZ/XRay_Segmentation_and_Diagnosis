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
root_dir = "/Users/riccardo/PycharmProjects/Lung_Segmentation"
data_dir = os.path.join(root_dir, "Data")
img_dir = os.path.join(data_dir, "ChestXRay/image")
mask_dir = os.path.join(data_dir, "ChestXRay/mask")
path_model = os.path.join(root_dir, "UnetXRay.pt")
# ---------------------------------------------

# Datasets - Train/Validation
# ---------------------------------------------
from Dataset_Class import XRayDataset
img_size = 256
train_set = XRayDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    size=(img_size, img_size),
    split="train",
    train_ratio=0.7
)
val_set = XRayDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    size=(img_size, img_size),
    split="val",
    train_ratio=0.7,
    shuffle=False
)
test_set = XRayDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    size=(img_size, img_size),
    split="test",
    train_ratio=0.7,
    shuffle=False
)

# Dataloaders
# ---------------------------------------------
batch_size = 8
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Model
# ---------------------------------------------
from Unet_Model import UNetXRay, count_params, get_lr
model = UNetXRay(
    in_channels=1,
    out_channels=1,
    encoder_depth=3,
    decoder_channels=[128, 64, 32],
).to(device)
print(model)
print(f"Model Parameters : {count_params(model)}")

# Hyperparameters
Loss = nn.BCELoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3,
                              weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       factor=0.8,
                                                       patience=2,
                                                       cooldown=1,
                                                       min_lr=0)
# Metrics
# -------------------------------------------
def accuracy_score(y_true, y_pred):
    """
    Calculates pixel-wise accuracy for binary segmentation.
    """
    # Flatten
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    # Correct predictions
    correct = torch.sum(y_true_f == y_pred_f)

    # Total pixels
    total = y_true_f.numel()

    return correct.float() / total

def jaccard_index(y_true, y_pred, smooth=100):
    """
    Calculates the Jaccard index (IoU).
    """
    # Flatten and cast
    y_true_f = y_true.float().view(-1)
    y_pred_f = y_pred.float().view(-1)

    # Intersection
    intersection = torch.sum(y_true_f * y_pred_f)

    # Union
    total = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection

    return (intersection + smooth) / (total + smooth)


def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Calculates the Dice coefficient.
    """
    # Flatten and cast
    y_true_f = y_true.float().view(-1)
    y_pred_f = y_pred.float().view(-1)

    # Intersection
    intersection = torch.sum(y_true_f * y_pred_f)

    # Dice score
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


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
    train_acc  = 0.0
    train_iou  = 0.0
    train_dice = 0.0
    total_samples = 0

    # Train loader iteration
    for _, data in enumerate(tqdm(train_loader)):
        x, y = data
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(x)  # shape : [B,C,S,S]

        # Compute Loss
        bce = Loss(out, y)
        bce.backward()

        optimizer.step()
        train_loss += bce.item()

        # Compute Metrics
        preds = (out > 0.5).long().cpu()
        targets = y.long().cpu()

        acc = accuracy_score(targets, preds)
        iou = jaccard_index(targets, preds)
        dice = dice_coefficient(targets, preds)

        train_acc += acc.item() * batch_size
        train_iou += iou.item() * batch_size
        train_dice += dice.item() * batch_size

        total_samples += batch_size

    train_loss /= total_samples
    train_acc  /= total_samples
    train_iou  /= total_samples
    train_dice /= total_samples

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc  = 0.0
    val_iou  = 0.0
    val_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader)):
            x, y = data
            batch_size = x.size(0)
            x, y = x.to(device), y.to(device)

            # Forward pass
            out = model(x)  # shape : [B,C,S,S]

            val_loss += Loss(out, y).item()

            # Compute Metrics
            preds = (out > 0.5).long().cpu()
            targets = y.long().cpu()

            acc = accuracy_score(targets, preds)
            iou = jaccard_index(targets, preds)
            dice = dice_coefficient(targets, preds)

            val_acc += acc.item() * batch_size
            val_iou += iou.item() * batch_size
            val_dice += dice.item() * batch_size

            total_samples += batch_size

    val_loss /= total_samples
    val_acc  /= total_samples
    val_iou  /= total_samples
    val_dice /= total_samples

    # --------------------------------
    history["lrs"].append(get_lr(optimizer))
    history["train_losses"].append(train_loss)
    history["val_losses"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["train_iou"].append(train_iou)
    history["val_iou"].append(val_iou)
    history["train_dice"].append(train_dice)
    history["val_dice"].append(val_dice)

    t2 = default_timer()
    print(f"Epoch : {epoch}, "
          f"| Train Loss : {np.round(train_loss, 4)}, "
          f"| Train Accuracy : {np.round(train_acc, 2)}, "
          f"| Train IoU : {np.round(train_iou, 2)}, "
          f"| Train Dice : {np.round(train_dice, 2)} "
          f"| LR : {get_lr(optimizer)}")
    print(f"Val Loss : {np.round(val_loss, 4)}, "
          f"| Val Accuracy : {np.round(val_acc, 2)}, "
          f"| Val IoU : {np.round(val_iou, 2)}, "
          f"| Val Dice : {np.round(val_dice, 2)}"
          f"| Time : {np.round(t2 - t1, 2)}")

    cur_val_loss = val_loss
    if cur_val_loss < min_loss:
        print('Loss Decreasing.. {:.5f} >> {:.5f} '.format(min_loss, cur_val_loss))
        min_loss = cur_val_loss
        best_model = copy.deepcopy(model.state_dict())
        patience = 0

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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle(model_name, fontsize=16)

    # --- Loss ---
    axes[0, 0].plot(history["train_losses"], label="Train")
    axes[0, 0].plot(history["val_losses"], label="Val")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid()

    # --- Accuracy ---
    axes[0, 1].plot(history["train_acc"], label="Train")
    axes[0, 1].plot(history["val_acc"], label="Val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid()

    # --- IoU ---
    axes[1, 0].plot(history["train_iou"], label="Train")
    axes[1, 0].plot(history["val_iou"], label="Val")
    axes[1, 0].set_title("IoU (Jaccard)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU")
    axes[1, 0].legend()
    axes[1, 0].grid()

    # --- Dice ---
    axes[1, 1].plot(history["train_dice"], label="Train")
    axes[1, 1].plot(history["val_dice"], label="Val")
    axes[1, 1].set_title("Dice Coefficient")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Dice")
    axes[1, 1].legend()
    axes[1, 1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Results", bbox_inches="tight")
    plt.show()

plot_results(history, "UnetXRay")
