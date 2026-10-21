"""PCam HDF5 dataset loading and image preprocessing."""
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 96


def image_transform(train=False):
    ops = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
    ops.extend([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    return transforms.Compose(ops)


class H5PcamDataset(Dataset):
    """Read matching official PCam *_x.h5 and *_y.h5 files lazily."""
    def __init__(self, images_file, labels_file, transform=None):
        self.images_file, self.labels_file = str(images_file), str(labels_file)
        self.transform = transform or image_transform()
        with h5py.File(self.images_file, "r") as f:
            self.length = len(f["x"])
        with h5py.File(self.labels_file, "r") as f:
            if len(f["y"]) != self.length:
                raise ValueError("Image and label files have different lengths")
        self._images = self._labels = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self._images is None:
            self._images = h5py.File(self.images_file, "r")["x"]
            self._labels = h5py.File(self.labels_file, "r")["y"]
        image = Image.fromarray(np.asarray(self._images[index]).astype(np.uint8))
        label = int(np.asarray(self._labels[index]).reshape(-1)[0])
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def find_h5_split(raw_dir, split):
    matches = list(Path(raw_dir).glob(f"*split_{split}_x.h5"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one *split_{split}_x.h5 in {raw_dir}; found {len(matches)}")
    x_file = matches[0]
    y_file = x_file.with_name(x_file.name[:-5] + "_y.h5")
    if not y_file.exists():
        raise FileNotFoundError(f"Matching label file not found: {y_file}")
    return x_file, y_file
