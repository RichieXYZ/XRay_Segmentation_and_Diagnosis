# Image Processing Libraries
import cv2
from cv2 import imread,resize
from scipy.ndimage import label, find_objects

# Data Handling Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# File and Operating System Libraries
import os

# Warnings Management
import warnings
warnings.filterwarnings('ignore')

# PATHS
root_dir = ""  # specify root path : "~/Xray_Segmentation_and_Diagnosis"
data_dir = os.path.join(root_dir, "Data")
img_dir  = os.path.join(data_dir, "ChestXRay/image")
mask_dir = os.path.join(data_dir, "ChestXRay/mask")

# Metadata .csv file
metadata = pd.read_csv(os.path.join(data_dir, "MetaData.csv"))
metadata.info()
print(metadata.head())

def get_colored_mask(image, mask_image, color=[255, 20, 255]):
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(mask_image, mask_image, mask=mask_image_gray)
    mask_coord = np.where(mask != [0, 0, 0])
    mask[mask_coord[0], mask_coord[1], :] = color
    ret = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
    return ret

# Plot some images examples
filenames = next(os.walk(img_dir))[2][:50]
print(filenames)
for file in filenames:
    img = imread(os.path.join(img_dir, file))
    msk = imread(os.path.join(mask_dir, file))

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.subplot(132)
    plt.title("Mask")
    plt.imshow(msk, cmap='binary_r')
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.subplot(133)
    plt.title("Image + Mask")
    plt.imshow(get_colored_mask(img, msk))
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.tight_layout()
    plt.show()

# Plot mask distribution heatmap
filenames = next(os.walk(img_dir))[2]
heatmap = np.zeros((1024, 1024))
for file in tqdm(filenames, total=len(filenames)):
    msk = imread(os.path.join(mask_dir, file), 0)
    msk = resize(msk, (1024, 1024))
    heatmap += msk
heatmap /= len(filenames)

plt.figure(figsize=(12, 12))

plt.imshow(heatmap, cmap='coolwarm')
plt.yticks([])
plt.xticks([])

plt.show()
