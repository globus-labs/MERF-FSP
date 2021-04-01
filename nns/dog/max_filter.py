import matplotlib.pyplot as plt
from scipy import ndimage as ndi, misc
from skimage import img_as_float, data
from skimage.feature import peak_local_max
from skimage.feature.peak import _get_high_intensity_peaks, _exclude_border
from skimage.io import imread
from torch import nn
import numpy as np
import torch

from sk_image.preprocess import make_figure, make_hist

im = img_as_float(
    imread("coins.jpg", as_gray=True)
)
# image_max is the dilation of im with a 20*20 structuring element
# It is used within peak_local_max function
min_distance = 20
exclude_border = True
footprint = None
size = 2 * min_distance + 1
m = nn.MaxPool2d(size, stride=1, padding=min_distance)

t_im = torch.from_numpy(im).unsqueeze(0)
# padded_t_im = nn.ConstantPad2d(min_distance, 0)(t_im)
image_max_t = m(t_im).detach().numpy()[0]
image_max = ndi.maximum_filter(im, size=size, mode="constant")

print((im - t_im[0].detach().numpy()).any())
print(np.abs(image_max - image_max_t).any())

# get_peak_mask
mask = im == image_max_t
threshold_abs = im.min()
threshold_rel = None
if threshold_rel is not None:
    threshold = max(threshold_abs, threshold_rel * im.max())
else:
    threshold = threshold_abs
print(threshold)
mask &= im > threshold
exclude_border = size if exclude_border else 0
mask = _exclude_border(mask, footprint, exclude_border)

coordinates = _get_high_intensity_peaks(im, mask, np.inf)

# Comparison between image_max and im to find the coordinates of local maxima

# display results
fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Original")

ax[1].imshow(image_max, cmap=plt.cm.gray)
ax[1].axis("off")
ax[1].set_title("Max filter")

coordinates = peak_local_max(im, min_distance=20)
ax[2].imshow(im, cmap=plt.cm.gray)
ax[2].autoscale(False)
ax[2].plot(coordinates[:, 1], coordinates[:, 0], "r.")
ax[2].axis("off")
ax[2].set_title("Peak local max")

fig.tight_layout()

plt.show()
