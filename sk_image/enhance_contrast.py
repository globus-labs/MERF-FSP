import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats, ndimage
from skimage import exposure, img_as_float
from skimage.filters import gaussian
from skimage.io import imread

from sk_image.preprocess import make_figure, show, major

AUTO_THRESHOLD = 5000


@dataclass
class ImageStats:
    @dataclass
    class Histogram:
        bin_counts: np.ndarray
        bin_centers: np.ndarray

    pixel_count: int
    histogram: Histogram

    mean: float
    mode: float
    std_dev: float

    min: int
    max: int

    n_bins: int
    bin_width: float

    def __init__(self, image, n_hist_bins=256):
        self.n_bins = n_hist_bins

        px, py = image.shape
        self.pixel_count = px * py
        self.histogram = self.Histogram(*exposure.histogram(image, nbins=n_hist_bins))

        self.mean = image.mean()
        self.mode = stats.mode(image, axis=None).mode.item()
        self.std_dev = image.std()

        self.min = image.min()
        self.max = image.max()
        self.bin_width = (self.max - self.min) / self.n_bins


def get_min_and_max(stats: ImageStats, saturation=0.0):
    if saturation > 0:
        threshold = stats.pixel_count * saturation / 200
    else:
        threshold = 0

    count = 0
    for bin_count, bin_center in zip(
        stats.histogram.bin_counts, stats.histogram.bin_centers
    ):
        count += bin_count
        found = count > threshold
        if found:
            break

    h_min = bin_center

    count = 0
    for bin_count, bin_center in reversed(
        list(zip(stats.histogram.bin_counts, stats.histogram.bin_centers))
    ):
        count += bin_count
        found = count > threshold
        if found:
            break

    h_max = bin_center
    return h_min, h_max


def stretch_composite_histogram(image, saturation=0.35):
    stats = ImageStats(image)
    h_min, h_max = get_min_and_max(stats, saturation)
    return exposure.rescale_intensity(image, in_range=(h_min, h_max), out_range="image")


def test_single():
    data_pth = Path("RawData")
    image_fn = Path("R-233_5-8-6_000117.T000.D000.P000.H000.PLIF1.TIF")
    image_pth = data_pth / image_fn
    print(image_pth)
    img_org = imread(image_pth)
    filtered_img = gaussian(img_org, sigma=1)
    # show(filtered_img)
    log_img = np.log(filtered_img)
    stats = ImageStats(log_img)
    log_img[log_img < stats.mode] = 0
    show(log_img)


def gradient_map(img):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # Define kernel for y differences
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # Perform x convolution
    x = ndimage.convolve(img, kx)
    # Perform y convolution
    y = ndimage.convolve(img, ky)
    sobel = np.hypot(x, y)
    return sobel


def test_multi():
    for image_pth in glob.glob("data/RawData/*.TIF"):
        img_org = imread(image_pth)
        maj_img = major(img_org)
        make_figure(maj_img, dpi=300)

def print_ranges():
    for image_pth in glob.glob("/Users/maksim/dev_projects/merf/data/RawData/**/*.TIF"):
        img_org = imread(image_pth)
        img_org = img_as_float(img_org)
        print(img_org.min(), img_org.max())



if __name__ == "__main__":
    # test_multi()
    # test_single()
    print_ranges()
