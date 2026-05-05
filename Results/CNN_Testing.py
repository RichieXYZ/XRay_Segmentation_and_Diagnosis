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

device = torch.device("cpu")

# PATHS
# ---------------------------------------------
root_dir = "/Users/riccardo/PycharmProjects/Lung_Segmentation"
data_dir = os.path.join(root_dir, "Data")
img_dir = os.path.join(data_dir, "ChestXRay/image")
mask_dir = os.path.join(data_dir, "ChestXRay/mask")
metadata_path = os.path.join(data_dir, "MetaData.csv")
ckpt_path = os.path.join(root_dir, "TBC_detection/TBC_Classifier_v2.pt")
# ---------------------------------------------

# Datasets - Testing
# ---------------------------------------------
from TBC_Dataset import TBCDataset
img_size = 256
test_set = TBCDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    metadata_path=metadata_path,
    size=(img_size, img_size),
    split="test",
    train_ratio=0.7
)

# Dataloaders
# ---------------------------------------------
batch_size = 1
test_loader   = DataLoader(test_set, batch_size=batch_size, shuffle=False)

Loss = nn.BCEWithLogitsLoss()

# Model
# ---------------------------------------------
from TBC_Classifier import TBC_Classifier_v2, count_params, get_lr
#enc_ch = [32, 64, 128]
#enc_ch = [128, 64, 32]
enc_ch = [256, 128, 64]
model = TBC_Classifier_v2(
    in_channels=1,
    encoder_depth=3,
    encoder_channels=enc_ch,
).to(device)
print(model)
print(f"Model Parameters : {count_params(model)}")

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

test_loss = 0.0
conf = torch.zeros(2, 2, dtype=torch.long)
total_samples = 0

with torch.no_grad():
    for _, data in enumerate(tqdm(test_loader)):
        x, y = data
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)

        out = model(x)
        y = y.float().unsqueeze(1)  # (B,) -> (B,1)
        test_loss += Loss(out, y).item()

        # Update confusion matrix as running inspection
        preds = (torch.sigmoid(out) > 0.5).long().squeeze(1).cpu()
        targets = y.long().squeeze(1).cpu()

        conf[0, 0] += ((preds == 0) & (targets == 0)).sum()
        conf[0, 1] += ((preds == 1) & (targets == 0)).sum()
        conf[1, 0] += ((preds == 0) & (targets == 1)).sum()
        conf[1, 1] += ((preds == 1) & (targets == 1)).sum()

        total_samples += batch_size

accuracy = (conf.diag().sum() / conf.sum()).item()
test_loss /= total_samples

print(f"Test Loss : {np.round(test_loss, 5)}, "
      f"| Test Accuracy : {np.round(accuracy, 3)}, ")
print("Confusion Matrix :")
print(conf.cpu().numpy())