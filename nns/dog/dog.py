import math
import numbers

import numpy as np
import torch
from skimage import img_as_float
from skimage.feature import peak_local_max
from skimage.feature.blob import _prune_blobs
from skimage.filters import gaussian
from skimage.io import imread
from torch import nn
from torch.nn import functional as F

from sk_image.blob import make_circles_fig
from sk_image.enhance_contrast import stretch_composite_histogram
from sk_image.preprocess import make_figure

DEBUG = True


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, *, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input: torch.Tensor):
        return self.conv(input, weight=self.weight, groups=self.groups)


class DifferenceOfGaussians(nn.Module):
    def __init__(self, *, max_sigma=10, min_sigma=1, sigma_ratio=1.2, truncate=4.0):
        super().__init__()
        #     truncate : float, optional
        #         Truncate the filter at this many standard deviations.
        #         Default is 4.0.

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        self.k = int(np.log(max_sigma / min_sigma) / np.log(sigma_ratio) + 1)
        # a geometric progression of standard deviations for gaussian kernels
        self.sigma_list = np.array(
            [min_sigma * (sigma_ratio ** i) for i in range(self.k + 1)]
        )
        print("sigmas: ", self.sigma_list)
        # max is performed in order to accommodate largest filter
        self.max_radius = int(truncate * max(self.sigma_list) + 0.5)
        self.gaussian_pyramid = nn.Conv2d(
            1, self.k + 1, 2 * self.max_radius + 1, bias=False
        )
        for i, s in enumerate(self.sigma_list):
            radius = int(truncate * s + 0.5)
            kernel = GaussianSmoothing(
                channels=1, kernel_size=2 * radius + 1, sigma=s
            ).weight.data[0]
            pad_size = self.max_radius - radius
            if pad_size > 0:
                padded_kernel = nn.ConstantPad2d(pad_size, 0)(kernel)
            else:
                padded_kernel = kernel
            self.gaussian_pyramid.weight.data[i].copy_(padded_kernel)
            if DEBUG:
                print(self.gaussian_pyramid.weight.data[i])

        for p in self.gaussian_pyramid.parameters():
            p.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded_input = nn.ConstantPad2d(self.max_radius, 0)(input)
        gaussian_images = self.gaussian_pyramid(padded_input)
        if DEBUG:
            print(gaussian_images.shape)
            for i, d in enumerate(gaussian_images[0]):
                print(d)
                make_figure(d.detach().numpy()).savefig(
                    f"/Users/maksim/dev_projects/merf/data/processed_images/debug/dog_gaussian_{i}.png"
                )
        # computing difference between two successive Gaussian blurred images
        # multiplying with average standard deviation provides scale invariance
        dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
            self.sigma_list[: self.k][np.newaxis, np.newaxis].T
        )
        if DEBUG:
            for i, d in enumerate(dog_images):
                print(d)
                make_figure(d.detach().numpy()).savefig(
                    f"/Users/maksim/dev_projects/merf/data/processed_images/debug/dog_dog_{i}.png"
                )
        return dog_images


def torch_dog(
    img_tensor, threshold=0.001, overlap=0.8, min_sigma=1, max_sigma=30, sigma_ratio=1.2
):
    # max_sigma = 10, min_sigma = 1, threshold = 0.02, overlap = 0.8
    with torch.no_grad():
        dog = DifferenceOfGaussians(
            min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio
        )
        dog.eval()
        dogs = dog(img_tensor)
        # permute to match local peaks api
        image_cube = dogs.permute(1, 2, 0).detach().numpy()

    ndim = 2
    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        footprint=np.ones((3,) * (ndim + 1)),
        threshold_rel=0.0,
    )
    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))

    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)

    # translate final column of lm, which contains the index of the
    # sigma that produced the maximum intensity value, into the sigma
    sigmas_of_peaks = dog.sigma_list[local_maxima[:, -1]]
    sigmas_of_peaks = np.expand_dims(sigmas_of_peaks, axis=1)

    # Remove sigma index and replace with sigmas
    lm = np.hstack([lm[:, :-1], sigmas_of_peaks])

    sigma_dim = sigmas_of_peaks.shape[1]
    return _prune_blobs(lm, overlap, sigma_dim=sigma_dim)


def torch_dog_test():
    with torch.no_grad():
        input = torch.rand(1, 1, 1000, 1000)
        input[0, 0, 400:600, 400:600] = 1
        make_figure(input.detach().numpy()[0][0]).show()
        dogs = torch_dog(input)
        for d in dogs:
            make_figure(d.detach().numpy()).show()


def torch_dog_img_test():
    image_pth = "/Users/maksim/dev_projects/merf/simulation/screenshot.png"

    img_orig = imread(image_pth, as_gray=True)
    make_figure(img_orig).show()
    # values have to be float and also between 0,1 for peak finding to work
    img_orig = img_as_float(img_orig)
    filtered_img = gaussian(img_orig, sigma=1)
    s2 = stretch_composite_histogram(filtered_img)
    make_figure(s2).show()
    t_image = torch.from_numpy(s2).float().unsqueeze(0).unsqueeze(0)
    blobs = torch_dog(t_image)
    blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
    print("blobs: ", len(blobs))
    make_circles_fig(s2, blobs).show()


if __name__ == "__main__":
    # torch_dog_test()
    torch_dog_img_test()
