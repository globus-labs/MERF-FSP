import math
import numbers
import os
import time
from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from opt_einsum import contract
from scipy import spatial
from skimage.feature.blob import _blob_overlap
from torch import nn
from torch.utils.data import DataLoader

from nn_dog.data import SimulPLIF

# noinspection PyUnresolvedReferences
from sk_image.preprocess import make_figure


BENCHMARK_TIMES = False


def close_pool(pool):
    pool.close()
    pool.terminate()
    pool.join()


class DifferenceOfGaussiansStandardConv(nn.Module):
    def __init__(
        self,
        *,
        max_sigma=10,
        min_sigma=1,
        sigma_bins=50,
        truncate=5.0,
        footprint=3,
        threshold=0.001,
        prune=True,
        overlap=0.5,
    ):
        super().__init__()

        self.footprint = footprint
        self.prune = prune
        self.overlap = overlap
        self.threshold = threshold

        self.sigma_list = np.concatenate(
            [
                np.linspace(min_sigma, max_sigma, sigma_bins),
                [max_sigma + (max_sigma - min_sigma) / (sigma_bins - 1)],
            ]
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)

        # max is performed in order to accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        self.gaussian_pyramid = nn.Conv2d(
            1,  # greyscale input
            sigma_bins + 1,  # sigma+1 filters so that there are sigma dogs
            2 * self.max_radius
            + 1,  # conv stack should be as wide as widest gaussian filter
            bias=False,
            padding=self.max_radius,  # hence no shrink of image
            padding_mode="zeros",
        )

        for i, s in enumerate(sigmas):
            radius = int(truncate * s + 0.5)
            kernel = torch_gaussian_kernel(width=2 * radius + 1, sigma=s.item())
            pad_size = self.max_radius - radius
            if pad_size > 0:
                padded_kernel = nn.ConstantPad2d(pad_size, 0)(kernel)
            else:
                padded_kernel = kernel
            self.gaussian_pyramid.weight.data[i].copy_(padded_kernel)

        self.padding = (self.footprint - 1) // 2
        if not isinstance(self.footprint, int):
            self.footprint = tuple(self.footprint)
            self.padding = tuple(self.padding)

        self.max_pool = nn.MaxPool3d(
            kernel_size=self.footprint, padding=self.padding, stride=1
        )

    def forward(self, input: torch.Tensor):
        gaussian_images = self.gaussian_pyramid(input)
        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )
        local_maxima = self.max_pool(dog_images.unsqueeze(0)).squeeze(0)
        mask = (local_maxima == dog_images) & (dog_images > self.threshold)
        return mask, local_maxima

    def make_blobs(self, mask, local_maxima=None):
        if local_maxima is not None:
            local_maxima = local_maxima[mask].detach().cpu().numpy()
        coords = mask.nonzero().cpu().numpy()
        cds = coords.astype(np.float64)
        # translate final column of cds, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = self.sigma_list[coords[:, 0]]
        # Remove sigma index and replace with sigmas
        cds = np.hstack([cds[:, 1:], sigmas_of_peaks[np.newaxis].T])
        if self.prune:
            blobs = prune_blobs(
                blobs_array=cds,
                overlap=self.overlap,
                local_maxima=local_maxima,
                sigma_dim=1,
            )
        else:
            blobs = cds

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


class DifferenceOfGaussiansFFT(nn.Module):
    """DoG filter.

    Apply a stack of gaussian filters to the input, take mutual differences,
    then find local maxima using a max filter heuristic.

    Attributes
    ----------
    threshold : minimum peak strength to consider
    prune: remove overlapping blobs
    overlap: minimum overlap between two blobs such that one will be pruned
    img_height: obv
    img_width: obv
    signal_ndim: last `signal_ndim` dimensions are the image
    sigma_list: [min_sigma, ... ,max_sigma] of length `sigma_bins`
    max_radius: maximum gaussian kernel radius (proportional to max_sigma)
    fft_height: fft size
    fft_width: fft size
    pad_input: padding functions that pads input out to fft size
    f_gaussian_pyramid: fourier space representation of gaussian kernel stack
    max_pool: max_pool filter for finding local maxima

    Parameters
    ----------
    min_sigma: minimum sigma that will be recognized
    max_sigma: maximum sigma that will be recognized
    sigma_bins: number of sigma between (inclusive) min/max sigma recognized
    truncate: width scale factor for the mesh for the gaussian kernels
    maxpool_footprint: maxpool kernel size. determines nearest blobs
    img_height:
    img_width:
    threshold:
    prune:
    overlap:

    Methods
    -------
    forward(input):
        forward pass i.e. perform the filtering. output blob center mask
    make_blobs(mask, local_maxima=None):
        make (x,y,r) from blob mask

    """

    def __init__(
        self,
        *,
        img_height: int,
        img_width: int,
        min_sigma: int = 1,
        max_sigma: int = 10,
        sigma_bins: int = 50,
        truncate: float = 4.0,
        maxpool_footprint: int = 3,
        threshold: float = 0.001,
        prune: bool = True,
        overlap: float = 0.5,
    ):
        super(DifferenceOfGaussiansFFT, self).__init__()
        self.prune = prune
        self.overlap = overlap
        self.threshold = threshold
        self.img_height = img_height
        self.img_width = img_width
        self.signal_ndim = 2

        self.sigma_list = np.concatenate(
            [
                np.linspace(min_sigma, max_sigma, sigma_bins),
                [max_sigma + (max_sigma - min_sigma) / (sigma_bins - 1)],
            ]
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)
        # print("gaussian pyramid sigmas: ", len(sigmas), sigmas)

        # accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        max_bandwidth = 2 * self.max_radius + 1
        # pad fft to prevent aliasing
        padded_height = img_height + max_bandwidth - 1
        padded_width = img_width + max_bandwidth - 1
        # round up to next power of 2 for cheaper fft.
        self.fft_height = 2 ** math.ceil(math.log2(padded_height))
        self.fft_width = 2 ** math.ceil(math.log2(padded_width))
        self.pad_input = nn.ConstantPad2d(
            (0, self.fft_width - img_width, 0, self.fft_height - img_height), 0
        )

        self.f_gaussian_pyramid = []
        kernel_pad = nn.ConstantPad2d(
            # left, right, top, bottom
            (0, self.fft_width - max_bandwidth, 0, self.fft_height - max_bandwidth),
            0,
        )
        for i, s in enumerate(sigmas):
            radius = int(truncate * s + 0.5)
            width = 2 * radius + 1
            kernel = torch_gaussian_kernel(width=width, sigma=s.item())

            # this is to align all of the kernels so that the eventual fft shifts a fixed amount
            center_pad_size = self.max_radius - radius
            if center_pad_size > 0:
                centered_kernel = nn.ConstantPad2d(center_pad_size, 0)(kernel)
            else:
                centered_kernel = kernel

            padded_kernel = kernel_pad(centered_kernel)

            f_kernel = torch.rfft(
                padded_kernel, signal_ndim=self.signal_ndim, onesided=True
            )
            self.f_gaussian_pyramid.append(f_kernel)

        self.f_gaussian_pyramid = nn.Parameter(
            torch.stack(self.f_gaussian_pyramid, dim=0), requires_grad=False
        )

        self.max_pool = nn.MaxPool3d(
            kernel_size=maxpool_footprint,
            padding=(maxpool_footprint - 1) // 2,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        img_height, img_width = list(input.size())[-self.signal_ndim :]
        assert (img_height, img_width) == (self.img_height, self.img_width)

        padded_input = self.pad_input(input)
        f_input = torch.rfft(padded_input, signal_ndim=self.signal_ndim, onesided=True)
        f_gaussian_images = comp_mul(self.f_gaussian_pyramid, f_input)
        gaussian_images = torch.irfft(
            f_gaussian_images,
            signal_ndim=self.signal_ndim,
            onesided=True,
            signal_sizes=padded_input.shape[1:],
        )

        # fft induces a shift so needs to be undone
        gaussian_images = gaussian_images[
            :,  # batch dimension
            :,  # filter dimension
            self.max_radius : self.img_height + self.max_radius,
            self.max_radius : self.img_width + self.max_radius,
        ]

        # computing difference between two successive Gaussian blurred images
        # multiplying with standard deviation provides scale invariance
        dog_images = (gaussian_images[:, :-1] - gaussian_images[:, 1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )
        local_maxima = self.max_pool(dog_images)
        mask = (local_maxima == dog_images) & (dog_images > self.threshold)
        return mask, local_maxima

    def make_blobs(
        self, mask: torch.Tensor, local_maxima: torch.Tensor = None
    ) -> np.ndarray:
        """Make blobs from mask produced by forward pass

        Parameters
        ----------
        mask: nonzero peaks after filtering
        local_maxima: peak values at nonzero positions in the mask. optional

        Returns
        -------
        blobs: blobs in the image

        """

        if local_maxima is not None:
            local_maxima = local_maxima[mask].detach().cpu().numpy()
        coords = mask.nonzero().cpu().numpy()
        cds = coords.astype(np.float64)
        # translate final column of cds, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = self.sigma_list[coords[:, 0]]
        # Remove sigma index and replace with sigmas
        cds = np.hstack([cds[:, 1:], sigmas_of_peaks[np.newaxis].T])
        if self.prune:
            blobs = prune_blobs(
                blobs_array=cds,
                overlap=self.overlap,
                local_maxima=local_maxima,
                sigma_dim=1,
            )
        else:
            blobs = cds

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def dog(self_i_input):
    self, i, input = self_i_input

    if BENCHMARK_TIMES:
        torch.cuda.synchronize(i)
        start = time.monotonic()

    f_gaussian_pyramid = self.f_gaussian_pyramids[i]
    padded_input = self.pad_input(input)
    f_input = torch.rfft(padded_input, signal_ndim=self.signal_ndim, onesided=True)
    f_gaussian_images = comp_mul(f_gaussian_pyramid, f_input)
    gaussian_images = torch.irfft(
        f_gaussian_images,
        signal_ndim=self.signal_ndim,
        onesided=True,
        signal_sizes=padded_input.shape[1:],
    )
    g = gaussian_images[
        :,  # batch dimension
        :,  # filter dimension
        self.max_radius : self.img_height + self.max_radius,
        self.max_radius : self.img_width + self.max_radius,
    ]
    g = g.to(self.master_device, non_blocking=self.pin_memory)

    if BENCHMARK_TIMES:
        torch.cuda.synchronize(i)
        end = time.monotonic()
        print(i, end - start)
    return i, g


class DifferenceOfGaussiansFFTParallel(nn.Module):
    def __init__(
        self,
        *,
        img_height: int,
        img_width: int,
        num_gpus: int,
        pin_memory: bool,
        master_device: int,
        min_sigma: int = 1,
        max_sigma: int = 10,
        sigma_bins: int = 50,
        truncate: float = 5.0,
        maxpool_footprint: int = 3,
        threshold: float = 0.001,
        prune: bool = True,
        overlap: float = 0.5,
    ):
        super(DifferenceOfGaussiansFFTParallel, self).__init__()
        self.prune = prune
        self.overlap = overlap
        self.threshold = threshold
        self.img_height = img_height
        self.img_width = img_width
        self.signal_ndim = 2
        self.num_gpus = num_gpus
        self.pin_memory = pin_memory
        self.master_device = master_device

        self.sigma_list = np.concatenate(
            [
                np.linspace(min_sigma, max_sigma, sigma_bins),
                [max_sigma + (max_sigma - min_sigma) / (sigma_bins - 1)],
            ]
        )
        sigmas = torch.from_numpy(self.sigma_list)
        self.register_buffer("sigmas", sigmas)
        # print("gaussian pyramid sigmas: ", len(sigmas), sigmas)

        # accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        max_bandwidth = 2 * self.max_radius + 1
        # pad fft to prevent aliasing
        padded_height = img_height + max_bandwidth - 1
        padded_width = img_width + max_bandwidth - 1
        # round up to next power of 2 for cheaper fft.
        self.fft_height = 2 ** math.ceil(math.log2(padded_height))
        self.fft_width = 2 ** math.ceil(math.log2(padded_width))
        self.pad_input = nn.ConstantPad2d(
            (0, self.fft_width - img_width, 0, self.fft_height - img_height), 0
        )

        self.f_gaussian_pyramids = []
        kernel_pad = nn.ConstantPad2d(
            # left, right, top, bottom
            (0, self.fft_width - max_bandwidth, 0, self.fft_height - max_bandwidth),
            0,
        )
        for i, chunk_sigmas in enumerate(
            chunks(sigmas, math.ceil(len(self.sigma_list) / num_gpus))
        ):
            f_gaussian_pyramid = []
            for j, s in enumerate(chunk_sigmas):
                radius = int(truncate * s + 0.5)
                width = 2 * radius + 1
                kernel = torch_gaussian_kernel(width=width, sigma=s.item())

                # this is to align all of the kernels so that the eventual fft shifts a fixed amount
                center_pad_size = self.max_radius - radius
                if center_pad_size > 0:
                    centered_kernel = nn.ConstantPad2d(center_pad_size, 0)(kernel)
                else:
                    centered_kernel = kernel

                padded_kernel = kernel_pad(centered_kernel)

                f_kernel = torch.rfft(
                    padded_kernel, signal_ndim=self.signal_ndim, onesided=True
                )
                f_gaussian_pyramid.append(f_kernel)

            self.f_gaussian_pyramids.append(torch.stack(f_gaussian_pyramid, dim=0))

        self.max_pool = nn.MaxPool3d(
            kernel_size=maxpool_footprint,
            padding=(maxpool_footprint - 1) // 2,
            stride=1,
        )

    def move_kernels_to_gpu(self):
        for i, f_gaussian_pyramid in enumerate(self.f_gaussian_pyramids):
            self.f_gaussian_pyramids[i] = f_gaussian_pyramid.to(f"cuda:{i}")

    def forward(self, input: torch.Tensor, pool) -> Tuple[torch.Tensor, torch.Tensor]:
        if BENCHMARK_TIMES:
            torch.cuda.synchronize(self.master_device)
            start = time.monotonic()

        all_gaussian_images = list(
            pool.map(
                dog,
                [
                    (self, i, input.to(f"cuda:{i}", non_blocking=self.pin_memory))
                    for i in range(len(self.f_gaussian_pyramids))
                ],
            )
        )
        all_gaussian_images = [v for i, v in all_gaussian_images]

        if BENCHMARK_TIMES:
            torch.cuda.synchronize(self.master_device)
            end = time.monotonic()
            print("parallel", end - start)

            torch.cuda.synchronize(self.master_device)
            start = time.monotonic()

        all_gaussian_images = torch.cat(all_gaussian_images, dim=1)
        dog_images = (all_gaussian_images[:, :-1] - all_gaussian_images[:, 1:]) * (
            self.sigmas[:-1].unsqueeze(0).unsqueeze(0).T
        )
        local_maxima = self.max_pool(dog_images)
        mask = (local_maxima == dog_images) & (dog_images > self.threshold)

        if BENCHMARK_TIMES:
            torch.cuda.synchronize(self.master_device)
            end = time.monotonic()
            print("rest", end - start)

        return mask, local_maxima

    def make_blobs(
        self, mask: torch.Tensor, local_maxima: torch.Tensor = None
    ) -> np.ndarray:
        """Make blobs from mask produced by forward pass

        Parameters
        ----------
        mask: nonzero peaks after filtering
        local_maxima: peak values at nonzero positions in the mask. optional

        Returns
        -------
        blobs: blobs in the image

        """

        if local_maxima is not None:
            local_maxima = local_maxima[mask].detach().cpu().numpy()
        coords = mask.nonzero().cpu().numpy()
        cds = coords.astype(np.float64)
        # translate final column of cds, which contains the index of the
        # sigma that produced the maximum intensity value, into the sigma
        sigmas_of_peaks = self.sigma_list[coords[:, 0]]
        # Remove sigma index and replace with sigmas
        cds = np.hstack([cds[:, 1:], sigmas_of_peaks[np.newaxis].T])
        if self.prune:
            blobs = prune_blobs(
                blobs_array=cds,
                overlap=self.overlap,
                local_maxima=local_maxima,
                sigma_dim=1,
            )
        else:
            blobs = cds

        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def torch_gaussian_kernel(
    width: int = 21, sigma: int = 3, dim: int = 2
) -> torch.Tensor:
    """Gaussian kernel

    Parameters
    ----------
    width: bandwidth of the kernel
    sigma: std of the kernel
    dim: dimensions of the kernel (images -> 2)

    Returns
    -------
    kernel : gaussian kernel

    """

    if isinstance(width, numbers.Number):
        width = [width] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid(
        [torch.arange(size, dtype=torch.float32) for size in width]
    )
    for size, std, mgrid in zip(width, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= (
            1
            / (std * math.sqrt(2 * math.pi))
            * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return kernel


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def comp_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Complex multiplies two complex 3d tensors

    x = (x_real, x_im)
    y = (y_real, y_im)
    x*y = (x_real*y_real - x_im*y_im, x_real*y_im + x_im*y_real)

    Last dimension is x2 with x[..., 0] real and x[..., 1] complex.
    Dimensions (-3,-2) must be equal of both a and b must be the same.

    Examples
    ________
    >>> f_filters = torch.rand((20, 1024, 1024, 2))
    >>> f_imgs = torch.rand((5, 1024, 1024, 2))
    >>> f_filtered_imgs = comp_mul(f_filters, f_imgs)

    Parameters
    ----------
    x : Last dimension is (a,b) of a+ib
    y : Last dimension is (a,b) of a+ib

    Returns
    -------
    z : x*y

    """

    # hadamard product of every filter against every batch image
    op = partial(contract, "fuv,buv->bfuv")
    assert x.shape[-1] == y.shape[-1] == 2
    x_real, x_im = x.unbind(-1)
    y_real, y_im = y.unbind(-1)
    z = torch.stack(
        [op(x_real, y_real) - op(x_im, y_im), op(x_real, y_im) + op(x_im, y_real)],
        dim=-1,
    )
    return z


def prune_blobs(
    *,
    blobs_array: np.ndarray,
    overlap: float,
    local_maxima: np.ndarray = None,
    sigma_dim: int = 1,
) -> np.ndarray:
    """Find non-overlapping blobs

    Parameters
    ----------
    blobs_array: n x 3 where first two cols are x,y coords and third col is blob radius
    overlap: minimum area overlap in order to prune one of the blobs
    local_maxima: optional maxima values at peaks. if included then stronger maxima will be chosen on overlap
    sigma_dim: which column in blobs_array has the radius

    Returns
    -------
    blobs_array: non-overlapping blobs

    """

    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array

    for (i, j) in pairs:
        blob1, blob2 = blobs_array[i], blobs_array[j]
        blob_overlap = _blob_overlap(blob1, blob2, sigma_dim=sigma_dim)
        if blob_overlap > overlap:
            # if local maxima then pick stronger maximum
            if local_maxima is not None:
                if local_maxima[i] > local_maxima[j]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0
            # else take average
            else:
                blob2[-1] = (blob1[-1] + blob2[-1]) / 2
                blob1[-1] = 0

    return blobs_array[blobs_array[:, -1] > 0]


def main_parallel():
    with torch.no_grad():
        image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
            "../simulation/screenshot.png"
        )

        # image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        #     "../data/RawData/R109_60deg_6-8-25_OHP-LS000252.T000.D000.P000.H000.PLIF1.TIF"
        # )
        screenshot = SimulPLIF(img_path=image_pth, num_repeats=1, load_truth=False)
        img_height, img_width = screenshot[0].squeeze(0).numpy().shape

        from nn_dog import PIN_MEMORY, NUM_GPUS, DEVICE

        train_dataloader = DataLoader(screenshot, batch_size=1, pin_memory=PIN_MEMORY)
        img_tensor = next(iter(train_dataloader))

        dog = DifferenceOfGaussiansFFTParallel(
            img_height=img_height,
            img_width=img_width,
            num_gpus=NUM_GPUS,
            pin_memory=PIN_MEMORY,
            master_device=DEVICE,
            sigma_bins=40,
            threshold=0.012,
            max_sigma=30,
        ).to(DEVICE, non_blocking=PIN_MEMORY)
        dog.move_kernels_to_gpu()
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()

        ts = []
        for i in range(100):
            torch.cuda.synchronize(DEVICE)
            start = time.monotonic()

            dog(img_tensor)

            torch.cuda.synchronize(DEVICE)
            end = time.monotonic()
            ts.append(end - start)
            print(end - start)
            print()
        print(sum(ts[10:]) / len(ts[10:]))
        # for m, l in zip(masks, local_maximas):
        #     blobs = dog.make_blobs(m, l)
        #     # make_circles_fig(screenshot[0].numpy(), blobs).savefig("blobs.png")
        #     break


def main():
    with torch.no_grad():
        # image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        #     "../simulation/screenshot.png"
        # )

        image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
            "../data/RawData/R109_60deg_6-8-25_OHP-LS000252.T000.D000.P000.H000.PLIF1.TIF"
        )
        screenshot = SimulPLIF(img_path=image_pth, num_repeats=1, load_truth=False)
        img_height, img_width = screenshot[0].squeeze(0).numpy().shape

        from nn_dog import PIN_MEMORY

        train_dataloader = DataLoader(screenshot, batch_size=1, pin_memory=PIN_MEMORY)
        img_tensor = next(iter(train_dataloader))

        dog = DifferenceOfGaussiansFFT(
            img_height=img_height, img_width=img_width, sigma_bins=20, threshold=0.012
        )
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
        masks, local_maximas = dog(img_tensor)
        for m, l in zip(masks, local_maximas):
            blobs = dog.make_blobs(m, l)
            make_circles_fig(screenshot[0].numpy(), blobs).show()


def test(i):
    time.sleep(i)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # main_parallel()
    POOL = mp.Pool(5)
    POOL.map(test, range(5))
    close_pool()

    POOL = mp.Pool(5)
    POOL.map(test, range(5))
    close_pool()
