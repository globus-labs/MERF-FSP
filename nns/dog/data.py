import csv
import glob
import os
from pathlib import Path

import numpy as np
import torch
from skimage import io, img_as_float
from skimage.filters import gaussian
from torch import nn
from torch.utils.data import Dataset

from sk_image.blob import make_circles_fig
from sk_image.enhance_contrast import stretch_composite_histogram


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class Trivial(Dataset):
    def __init__(
        self, img_path: Path, num_repeats=10, min_sigma=1, max_sigma=40, sigma_bins=50
    ):
        self.img_path = img_path
        self.repeats = num_repeats
        print(img_path.parent / "truth.csv")

        sigma_list = np.linspace(
            start=min_sigma,
            stop=max_sigma + (max_sigma - min_sigma) / sigma_bins,
            num=sigma_bins + 1,
        )
        image = io.imread(self.img_path, as_gray=True)
        self.truth = torch.zeros(*image.shape)
        with open(img_path.parent / "truth.csv") as csvfile:
            reader = csv.reader(csvfile)
            # skip header
            next(reader)
            for i, (x, y, _z, r) in enumerate(reader):
                x, y, r_idx = (
                    int(float(x)),
                    int(float(y)),
                    int(find_nearest(sigma_list, float(r))),
                )
                self.truth[y, x] = r_idx

    def transform(self, img):
        img = img_as_float(img)
        img = gaussian(img, sigma=1)
        img = stretch_composite_histogram(img)
        img = torch.from_numpy(img).float()
        return img

    def __len__(self):
        return self.repeats

    def __getitem__(self, idx):
        image = io.imread(self.img_path, as_gray=True)
        t_img = self.transform(image)
        # return t_img, self.truth.type(torch.LongTensor)
        return t_img


class PLIF(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, plif_dir: Path):
        self.img_paths = glob.glob(str(plif_dir / "*.TIF"))

    def transform(self, img):
        img = img_as_float(img)
        img = gaussian(img, sigma=1)
        img = stretch_composite_histogram(img)
        img = torch.from_numpy(img).float()
        return img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = io.imread(self.img_paths[idx], as_gray=True)
        t_img = self.transform(image)
        return t_img
