import math
import numbers
import os
import time
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
from opt_einsum import contract
from torch import nn
from torch.multiprocessing import set_start_method, Pool

from nn_dog.model import prune_blobs


class DifferenceOfGaussiansFFT(nn.Module):
    def __init__(
        self,
        *,
        img_height: int,
        img_width: int,
        min_sigma: int = 1,
        max_sigma: int = 10,
        sigma_bins: int = 50,
        truncate: float = 5.0,
    ):
        super(DifferenceOfGaussiansFFT, self).__init__()
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

        return gaussian_images


class MakeBlobs(nn.Module):
    def __init__(
        self,
        maxpool_footprint: int = 3,
        prune: bool = True,
        min_sigma: int = 1,
        max_sigma: int = 10,
        sigma_bins: int = 48,
        threshold: float = 0.001,
        overlap: float = 0.5,
    ):
        super(MakeBlobs, self).__init__()

        self.overlap = overlap
        self.threshold = threshold
        self.prune = prune

        self.max_pool = nn.MaxPool3d(
            kernel_size=maxpool_footprint,
            padding=(maxpool_footprint - 1) // 2,
            stride=1,
        )
        self.sigma_list = np.concatenate(
            [
                np.linspace(min_sigma, max_sigma, sigma_bins),
                [max_sigma + (max_sigma - min_sigma) / (sigma_bins - 1)],
            ]
        )
        self.register_buffer("sigmas", torch.from_numpy(self.sigma_list))

    def forward(self, gaussian_images):
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


def run(rank, size):
    sigma_bins = 24
    with torch.no_grad():
        img_tensor_cpu = torch.rand((1, 1000, 1000))

        dog = DifferenceOfGaussiansFFT(
            img_height=1000, img_width=1000, sigma_bins=sigma_bins // size, max_sigma=30
        ).to(rank)
        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()

        if rank == 0:
            make_blobs = MakeBlobs(sigma_bins=sigma_bins, max_sigma=30, prune=False).to(rank)
            for p in make_blobs.parameters():
                p.requires_grad = False
            make_blobs.eval()

        torch.cuda.synchronize(rank)

        if rank == 0:
            start = time.monotonic()
            s = torch.cuda.current_stream(rank)
            e_start = torch.cuda.Event(enable_timing=True)
            e_finish = torch.cuda.Event(enable_timing=True)
            s.record_event(e_start)

        img_tensor = img_tensor_cpu.to(rank)
        for i in range(10):
            gaussian_images = dog(img_tensor)
            gaussian_images = gaussian_images.contiguous()
            output = [gaussian_images.clone() for _ in range(size)]
            dist.all_gather(tensor_list=output, tensor=gaussian_images)
            gaussian_images = torch.cat(output, dim=1)
            if rank == 0:
                mask, local_maxima = make_blobs(gaussian_images[:,:sigma_bins+1])
                blobs = make_blobs.make_blobs(mask, local_maxima)

        if rank == 0:
            torch.cuda.synchronize(rank)
            s.record_event(e_finish)
            e_finish.synchronize()
            end = time.monotonic()

            print(
                f"rank {rank} Iteration forward latency is {e_start.elapsed_time(e_finish)}"
            )
            print("end - start = ", end - start)


def init_process(rank_size_fn, backend="nccl"):
    rank, size, fn = rank_size_fn
    """ Initialize the distributed environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    return fn(rank, size)


if __name__ == "__main__":
    set_start_method("spawn")

    size = 1
    print(f"\n====== size = {size}  ======\n")
    pool = Pool(processes=size)
    start = time.monotonic()
    res = pool.map(init_process, [(i, size, run) for i in range(size)])
    end = time.monotonic()
    print("wall time", end - start)
    pool.close()

    size = 2
    print(f"\n====== size = {size}  ======\n")
    pool = Pool(processes=size)
    start = time.monotonic()
    res = pool.map(init_process, [(i, size, run) for i in range(size)])
    end = time.monotonic()
    print("wall time", end - start)
    pool.close()

    size = 3
    print(f"\n====== size = {size}  ======\n")
    pool = Pool(processes=size)
    start = time.monotonic()
    res = pool.map(init_process, [(i, size, run) for i in range(size)])
    end = time.monotonic()
    print("wall time", end - start)
    pool.close()

    size = 4
    print(f"\n====== size = {size}  ======\n")
    pool = Pool(processes=size)
    start = time.monotonic()
    res = pool.map(init_process, [(i, size, run) for i in range(size)])
    end = time.monotonic()
    print("wall time", end - start)
    pool.close()

