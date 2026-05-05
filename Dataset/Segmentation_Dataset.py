# Image Processing Libraries
from cv2 import imread,resize

# Data Handling Libraries
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt

# File and Operating System Libraries
import os

# Warnings Management
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self,
                 img_dir,
                 mask_dir,
                 size=(512, 512),
                 split="train",
                 train_ratio=0.7,
                 seed=1,
                 shuffle=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # List image and mask files
        self.img_filenames  = sorted(os.listdir(self.img_dir))
        self.mask_filenames = sorted(os.listdir(self.mask_dir))
        print(f"Number of data samples : {len(self.img_filenames)}")

        # Ensure the number of images matches the number of masks
        assert len(self.img_filenames) == len(self.mask_filenames), \
            "The number of images and masks must be the same"

        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.split = split

        self.indexes = np.arange(len(self.img_filenames))  # Indices for shuffling

        # Splitting
        train_idx = int(np.floor(train_ratio * len(self.indexes)))
        val_idx = int(np.floor((1 + train_ratio) / 2 * len(self.indexes)))

        if self.split == "train":
            self.indexes = self.indexes[:train_idx]
        elif self.split == "val":
            self.indexes = self.indexes[train_idx:val_idx]
        elif self.split == "test":
            self.indexes = self.indexes[val_idx:]

        print(f"Number of {self.split} samples : {len(self.indexes)}")

        # If shuffle is enabled, shuffle the indices after each epoch
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(len(self.indexes))

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.img_filenames[index])
        img = imread(img_path, 0)  # Read image
        img = resize(img, self.size)  # Resize to target size
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize to [0, 1]

        # Load and preprocess mask
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[index])
        mask = imread(mask_path, 0)  # Read mask
        mask = resize(mask, self.size)  # Resize to target size
        # mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = mask / 255.0  # Normalize to [0, 1]
        mask[mask > 0.5] = 1  # Binary mask

        return torch.tensor(np.array(img), dtype=torch.float32), torch.tensor(np.array(mask), dtype=torch.float32)

    def on_epoch_end(self):
        """
        Shuffle the dataset after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)


if __name__=="__main__":
    # PATHS
    root_dir = ""
    data_dir = os.path.join(root_dir, "Data")
    img_dir = os.path.join(data_dir, "ChestXRay/image")
    mask_dir = os.path.join(data_dir, "ChestXRay/mask")

    # Metadata .csv file
    # metadata = pd.read_csv(os.path.join(data_dir, "MetaData.csv"))
    img_size = 256
    train_data = XRayDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        size=(img_size, img_size),
        split="train",
        train_ratio=0.7
    )
    val_data = XRayDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        size=(img_size, img_size),
        split="val",
        train_ratio=0.7
    )
    test_data = XRayDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        size=(img_size, img_size),
        split="test",
        train_ratio=0.7
    )

    print(train_data.indexes)
    print(val_data.indexes)
    print(test_data.indexes)

    train_data.on_epoch_end()
    print(train_data.indexes)

    img, msk = train_data.__getitem__(0)
    print(img.shape)
    print(msk.shape)

    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.imshow(img.squeeze(), cmap="Greys_r")
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.subplot(122)
    plt.imshow(msk.squeeze(), cmap='binary_r')
    plt.yticks([])
    plt.xticks([])
    plt.box(False)

    plt.tight_layout()
    plt.show()

