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
ckpt_path = os.path.join(root_dir, "TBC_detection/TBC_Classifier_v30.pt")
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
from TBC_Resnet import TBC_ResNet18, count_params, get_lr
model = TBC_ResNet18().to(device)
print(model)
print(f"Model Parameters : {count_params(model)}")

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x):
        self.model.zero_grad()

        out = self.model(x)
        out.backward(torch.ones_like(out))

        grads = self.gradients
        activations = self.activations

        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


def overlay_cam(img_tensor, cam):
    img = img_tensor.squeeze().cpu().numpy()

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    img = np.uint8(255 * img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    overlay = 0.6 * img + 0.4 * cam
    return np.uint8(overlay)

target_layer = model.layer4
gradcam = GradCAM(model, target_layer)

test_loss = 0.0
conf = torch.zeros(2, 2, dtype=torch.long)
total_samples = 0
max_visualizations = 10
vis_count = 0

for i, data in enumerate(tqdm(test_loader)):
    x, y = data
    batch_size = x.size(0)
    x, y = x.to(device), y.to(device)

    # --- Normal inference (no grad) ---
    with torch.no_grad():
        out = model(x)
        y = y.float().unsqueeze(1)
        test_loss += Loss(out, y).item()

        preds = (torch.sigmoid(out) > 0.5).long().squeeze(1).cpu()
        targets = y.long().squeeze(1).cpu()

        conf[0, 0] += ((preds == 0) & (targets == 0)).sum()
        conf[0, 1] += ((preds == 1) & (targets == 0)).sum()
        conf[1, 0] += ((preds == 0) & (targets == 1)).sum()
        conf[1, 1] += ((preds == 1) & (targets == 1)).sum()

        total_samples += batch_size

    # --- Grad-CAM (ONLY for few samples) ---
    if vis_count < max_visualizations:
        model.eval()

        x_cam = x.clone().detach().requires_grad_(True)

        cam = gradcam.generate(x_cam)

        overlay = overlay_cam(x[0], cam)

        pred_label = preds.item()
        true_label = targets.item()

        plt.figure(figsize=(4, 4))
        plt.imshow(overlay)
        plt.title(f"Pred: {pred_label} | True: {true_label}")
        plt.axis("off")
        plt.show()

        vis_count += 1

accuracy = (conf.diag().sum() / conf.sum()).item()
test_loss /= total_samples

print(f"Test Loss : {np.round(test_loss, 5)}, "
      f"| Test Accuracy : {np.round(accuracy, 3)}, ")
print("Confusion Matrix :")
print(conf.cpu().numpy())