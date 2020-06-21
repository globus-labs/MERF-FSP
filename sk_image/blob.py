import os
import time
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import color, morphology, img_as_float
from skimage.draw import circle_perimeter
from skimage.feature import canny, blob_dog, blob_log, blob_doh, peak_local_max
from skimage.feature.blob import _prune_blobs
from skimage.filters import sobel, gaussian, threshold_otsu
from skimage.io import imread
from skimage.transform import hough_circle, hough_circle_peaks

# from simulation.simulate import create_circular_mask

# Load picture and detect edges
from nn_dog.model import prune_blobs
from sk_image.enhance_contrast import stretch_composite_histogram


def cpu_blob_dog(
    image,
    min_sigma=1,
    max_sigma=50,
    sigma_bins=10,
    threshold=2.0,
    overlap=0.5,
    prune=True,
    *,
    exclude_border=False
):
    image = img_as_float(image)

    # if both min and max sigma are scalar, function returns only one sigma
    scalar_sigma = np.isscalar(max_sigma) and np.isscalar(min_sigma)

    # Gaussian filter requires that sequence-type sigmas have same
    # dimensionality as image. This broadcasts scalar kernels
    if np.isscalar(max_sigma):
        max_sigma = np.full(image.ndim, max_sigma, dtype=float)
    if np.isscalar(min_sigma):
        min_sigma = np.full(image.ndim, min_sigma, dtype=float)

    # Convert sequence types to array
    min_sigma = np.asarray(min_sigma, dtype=float)
    max_sigma = np.asarray(max_sigma, dtype=float)

    # k such that min_sigma*(sigma_ratio**k) > max_sigma
    # k = int(np.mean(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1))
    # a geometric progression of standard deviations for gaussian kernels
    # sigma_list = np.array([min_sigma * (sigma_ratio ** i) for i in range(k + 1)])

    sigma_list = np.concatenate(
        [
            np.linspace(min_sigma, max_sigma, sigma_bins),
            [max_sigma + (max_sigma - min_sigma) / (sigma_bins - 1)],
        ]
    )
    gaussian_images = [gaussian_filter(image, s) for s in sigma_list]

    # computing difference between two successive Gaussian blurred images
    # multiplying with average standard deviation provides scale invariance
    dog_images = [
        (gaussian_images[i] - gaussian_images[i + 1]) * np.mean(sigma_list[i])
        for i in range(sigma_bins)
    ]

    image_cube = np.stack(dog_images, axis=-1)

    # local_maxima = get_local_maxima(image_cube, threshold)
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3,) * (image.ndim + 1)),
        exclude_border=exclude_border,
    )
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = sigma_list[local_maxima[:, -1]]

    if scalar_sigma:
        # select one sigma column, keeping dimension
        sigmas_of_peaks = sigmas_of_peaks[:, 0:1]

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    if prune:
        return prune_blobs(blobs_array=lm, overlap=overlap, sigma_dim=sigma_dim)
    else:
        return lm


def circle(image):
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)

    # Detect two radii
    hough_radii = np.arange(20, 35, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=3
    )

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (220, 20, 20)

    ax.imshow(image, cmap=plt.cm.gray)
    plt.show()


def segmentation(image):
    elevation_map = sobel(image)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(elevation_map, cmap=plt.cm.gray)
    ax.set_title("elevation map")
    ax.axis("off")
    plt.show()
    markers = np.zeros_like(image)
    markers[image < 30] = 1
    markers[image > 150] = 2

    fig, ax = plt.subplots(figsize=(4 * 4, 3 * 4))
    ax.imshow(markers, cmap=plt.cm.nipy_spectral)
    ax.set_title("markers")
    ax.axis("off")
    plt.show()
    segmentation = morphology.watershed(elevation_map, markers)

    fig, ax = plt.subplots(figsize=(4 * 4, 3 * 4))
    ax.imshow(segmentation, cmap=plt.cm.gray)
    ax.set_title("segmentation")
    ax.axis("off")
    plt.show()

    return segmentation


def dog(image):
    blobs = blob_dog(image, max_sigma=10, min_sigma=1, threshold=0.005, overlap=0.8)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    return blobs


def log(image):
    blobs_log = blob_log(image, max_sigma=10, min_sigma=5, threshold=0.02, overlap=0.9)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    return blobs_log


def doh(image):
    blobs_doh = blob_doh(image, max_sigma=30, threshold=0.01)
    return blobs_doh


def make_circles_fig(image, blobs, title=None, dpi=96):
    px, py = image.shape
    fig = plt.figure(figsize=(py / numpy.float(dpi), px / numpy.float(dpi)))
    if title is None:
        dims = [0.0, 0.0, 1.0, 1.0]
    else:
        dims = [0.0, 0.0, 1.0, 0.95]
    ax = fig.add_axes(dims, yticks=[], xticks=[], frame_on=False)
    ax.imshow(image, cmap="gray")
    ax.set_title(title, fontsize=50)
    for y, x, r in blobs:
        c = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        ax.add_patch(c)
    return fig


def hough(img):
    pass


def area(img, blob):
    h, w = img.shape
    y, x, r = blob
    blob_mask = create_circular_mask(h, w, (x, y), r)
    img = img.copy()
    img[~blob_mask] = 0
    thresh = threshold_otsu(img)
    img[img >= thresh] = 1
    img[img <= thresh] = 0
    return np.sum(img)


def main():
    start = time.monotonic()
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    img_orig = imread(image_pth, as_gray=True)
    filtered_img = gaussian(img_orig, sigma=1)
    s2 = stretch_composite_histogram(filtered_img)

    blobs = blob_dog(
        s2, max_sigma=10, min_sigma=1, threshold=0.001, overlap=0.5, sigma_ratio=1.01
    )
    print("dog time ", time.monotonic() - start)
    blobs[:, 2] = blobs[:, 2] * sqrt(2)
    print(len(blobs))
    # make_circles_fig(s2, blobs).show()
    #
    # plt.hist([r for (_, _, r) in blobs], bins=256)
    # plt.show()
    # areas = []
    # for blob in blobs:
    #     areas.append(area(s2, blob))
    #
    # plt.hist(areas, bins=256)
    # plt.show()


if __name__ == "__main__":
    main()
