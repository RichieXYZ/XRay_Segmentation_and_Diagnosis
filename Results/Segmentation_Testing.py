# Warnings Management
import warnings
warnings.filterwarnings('ignore')

import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

device = torch.device("cpu")

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
from TBC_detection.Dataset_Class import XRayDataset
img_size = 256

test_set = XRayDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    size=(img_size, img_size),
    split="test",
    train_ratio=0.7,
    shuffle=False
)

# Model
# ---------------------------------------------
from Unet_Model import UNetXRay
model = UNetXRay(
    in_channels=1,
    out_channels=1,
    encoder_depth=3,
    decoder_channels=[128, 64, 32],
).to(device)

ckpt_path = "/Users/riccardo/PycharmProjects/Lung_Segmentation/UnetXRay.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model.eval()

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

def get_colored_mask(image, mask_image, color=[255, 20, 255]):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    mask_coord = np.where(mask != [0, 0, 0])
    mask[mask_coord[0], mask_coord[1], :] = color
    ret = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
    return ret


with torch.no_grad():
    for i in range(test_set.__len__()):
        img, msk = test_set.__getitem__(i)
        img, msk = img.unsqueeze(0), msk.unsqueeze(0)

        # Forward pass
        out = model(img)  # shape : [B,C,S,S]
        preds = (out > 0.5).long()
        targets = msk.long()

        acc = accuracy_score(targets, preds)
        iou = jaccard_index(targets, preds)
        dice = dice_coefficient(targets, preds)

        print(f"Item {i} | Accuracy = {acc:.3f} | IoU = {iou:.3f} | Dice = {dice:.3f}")

        img = (img.squeeze().numpy()*255).astype(np.uint8)
        msk = msk.squeeze().numpy().astype(np.uint8)
        preds = preds.squeeze().numpy().astype(np.uint8)

        # Convert grayscale image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2RGB)
        preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2RGB)

        plt.figure(figsize=(12, 4))

        plt.subplot(141)
        plt.imshow(img)
        plt.title('Image')
        plt.yticks([])
        plt.xticks([])
        plt.box(False)

        plt.subplot(142)
        plt.imshow(get_colored_mask(img, msk))
        plt.title('Mask (Actual)')
        plt.yticks([])
        plt.xticks([])
        plt.box(False)

        plt.subplot(143)
        plt.imshow(get_colored_mask(img, preds, color=[255, 30, 0]))
        plt.title('Mask (Prediction)')
        plt.yticks([])
        plt.xticks([])
        plt.box(False)

        plt.subplot(144)
        plt.imshow(msk-preds)
        plt.title('Error')
        plt.yticks([])
        plt.xticks([])
        plt.box(False)

        plt.tight_layout()
        plt.show()