# Image Processing Libraries
import cv2
from cv2 import imread,resize

# Data Handling Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Visualization Libraries
import matplotlib.pyplot as plt

# File and Operating System Libraries
import os

# Warnings Management
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import random

class TBCDataset(Dataset):
    def __init__(self,
                 img_dir,
                 mask_dir,
                 metadata_path,
                 size=(512, 512),
                 split="train",
                 train_ratio=0.7,
                 seed=1,
                 shuffle=True,
                 **kwargs):
        super().__init__(**kwargs)

        # Class Variables
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.metadata = pd.read_csv(metadata_path)
        self.metadata.drop(["gender", "age", "county", "remarks"], axis=1, inplace=True)

        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.split = split

        # Create index list
        self.indexes = np.arange(len(self.metadata))

        # Stratified split
        labels = self.metadata["ptb"].values

        train_idx, temp_idx = train_test_split(
            self.indexes,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=self.seed
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            stratify=labels[temp_idx],
            random_state=self.seed
        )

        if self.split == "train":
            self.indexes = train_idx
        elif self.split == "val":
            self.indexes = val_idx
        elif self.split == "test":
            self.indexes = test_idx

        print(f"Number of {self.split} samples : {len(self.indexes)}")
        # Verify class distribution in splits
        print(f"Class distribution : Negatives (0) = {np.bincount(self.metadata['ptb'][self.indexes])[0]} | "
              f"Positives (1) = {np.bincount(self.metadata['ptb'][self.indexes])[1]}")

    def __len__(self):
        return int(len(self.indexes))

    def __getitem__(self, index):
        item = self.indexes[index]
        filename = str(self.metadata['id'][item]) + ".png"
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        img = imread(img_path, 0)  # Read image
        img = resize(img, self.size)  # Resize to target size
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize to [0, 1]

        mask = imread(mask_path, 0)  # Read mask
        mask = resize(mask, self.size)  # Resize to target size
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask / 255.0  # Normalize to [0, 1]
        mask[mask > 0.5] = 1  # Binary mask

        masked_img = img*mask
        label = self.metadata['ptb'][item]

        return torch.tensor(np.array(masked_img), dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


import torchvision.transforms as T

class TBCDataset2(Dataset):
    def __init__(self,
                 img_dir,
                 mask_dir,
                 metadata_path,
                 size=(256, 256),
                 split="train",
                 train_ratio=0.7,
                 seed=1,
                 shuffle=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.metadata = pd.read_csv(metadata_path)
        self.metadata.drop(["gender", "age", "county", "remarks"], axis=1, inplace=True)

        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.split = split

        # --- Stratified split ---
        self.indexes = np.arange(len(self.metadata))
        labels = self.metadata["ptb"].values

        train_idx, temp_idx = train_test_split(
            self.indexes,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=self.seed
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            stratify=labels[temp_idx],
            random_state=self.seed
        )

        if self.split == "train":
            self.indexes = train_idx
        elif self.split == "val":
            self.indexes = val_idx
        elif self.split == "test":
            self.indexes = test_idx

        print(f"Number of {self.split} samples : {len(self.indexes)}")
        print(f"Class distribution : Neg (0) = {np.bincount(self.metadata['ptb'][self.indexes])[0]} | "
              f"Pos (1) = {np.bincount(self.metadata['ptb'][self.indexes])[1]}")

        # --- Augmentations ---
        if self.split == "train":
            self.base_transform = T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.base_transform = T.Compose([
                T.ToPILImage(),
            ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        item = self.indexes[index]
        filename = str(self.metadata['id'][item]) + ".png"

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # --- Load ---
        img = imread(img_path, 0)
        mask = imread(mask_path, 0)

        img = resize(img, self.size)
        mask = resize(mask, self.size)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)

        # Normalize
        #img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        img = img / 255.0
        mask = mask / 255.0
        mask[mask > 0.5] = 1

        # Apply mask
        mean = np.mean(img)
        masked_img = img * mask + (1 - mask) * mean

        # Convert to uint8 for PIL transforms
        masked_img = (masked_img * 255).astype(np.uint8)

        # Apply transform
        masked_img = self.base_transform(masked_img)

        if self.split == "train":
            # Sample parameters
            angle = random.uniform(-10, 10)

            max_dx = int(0.05 * masked_img.size[0])
            max_dy = int(0.05 * masked_img.size[1])
            translations = (
                random.randint(-max_dx, max_dx),
                random.randint(-max_dy, max_dy)
            )

            scale = random.uniform(0.95, 1.05)

            # Apply affine with dynamic fill
            masked_img = F.affine(
                masked_img,
                angle=angle,
                translate=translations,
                scale=scale,
                shear=0,
                fill=mean*255
            )

            # Optional: crop after affine
            masked_img = F.resized_crop(
                masked_img,
                top=0,
                left=0,
                height=masked_img.size[1],
                width=masked_img.size[0],
                size=self.size
            )
        else:
            masked_img = F.resize(masked_img, self.size)

        # Convert to tensor
        masked_img = self.to_tensor(masked_img)

        label = self.metadata['ptb'][item]

        return masked_img, torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    # PATHS
    root_dir = ""  
    data_dir = os.path.join(root_dir, "Data")
    img_dir = os.path.join(data_dir, "ChestXRay/image")
    mask_dir = os.path.join(data_dir, "ChestXRay/mask")
    metadata_path = os.path.join(data_dir, "MetaData.csv")

    metadata = pd.read_csv(metadata_path)
    print(metadata.head())

    img_size = 256
    train_data = TBCDataset2(
        img_dir=img_dir,
        mask_dir=mask_dir,
        metadata_path=metadata_path,
        size=(img_size, img_size),
        split="train",
        train_ratio=0.7
    )
    val_data = TBCDataset2(
        img_dir=img_dir,
        mask_dir=mask_dir,
        metadata_path=metadata_path,
        size=(img_size, img_size),
        split="val",
        train_ratio=0.7
    )
    test_data = TBCDataset2(
        img_dir=img_dir,
        mask_dir=mask_dir,
        metadata_path=metadata_path,
        size=(img_size, img_size),
        split="test",
        train_ratio=0.7
    )
    #print(train_data.indexes)
    #print(val_data.indexes)
    #print(test_data.indexes)

    for i in range(len(train_data.indexes)):
        img, label = train_data.__getitem__(0)
        print(img.shape)
        print(label)

        title = "Positive (1)" if int(label) == 1 else "Negative (0)"
        plt.figure(figsize=(15, 5))
        plt.title(title)
        plt.imshow(img.squeeze(), cmap="Greys_r")
        plt.yticks([])
        plt.xticks([])
        plt.box(False)

        plt.tight_layout()
        plt.show()
