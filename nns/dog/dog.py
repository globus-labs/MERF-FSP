import math
import numbers
import os
from os.path import expanduser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from skimage.feature.blob import _prune_blobs
from torch import nn
from torch.multiprocessing import set_start_method

from nns.dog.data import Trivial, PLIF
from sk_image.blob import make_circles_fig
from sk_image.preprocess import make_figure

DATA_DIR = os.environ.get("FSP_DATA_DIR")
if DATA_DIR is None:
    raise Exception("need to specify env var FSP_DATA_DIR")
DATA_DIR = Path(expanduser(DATA_DIR))
NUM_GPUS = torch.cuda.device_count()
print(f"num gpus: {NUM_GPUS}")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def torch_gaussian_kernel(width=21, sigma=3, dim=2):
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
            * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)
    return kernel


class DifferenceOfGaussians(nn.Module):
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
        self.threshold = threshold
        self.prune = prune
        self.overlap = overlap

        sigma_ratio = 1 + np.log(max_sigma / min_sigma) / sigma_bins
        # a geometric progression of standard deviations for gaussian kernels
        self.sigma_list = np.asarray(
            [min_sigma * (sigma_ratio ** i) for i in range(sigma_bins + 1)]
        )
        sigmas = torch.from_numpy(self.sigma_list)
        # print("gaussian pyramid sigmas: ", len(sigmas), sigmas)
        # max is performed in order to accommodate largest filter
        self.max_radius = int(truncate * max(sigmas) + 0.5)
        self.padding = (self.footprint - 1) // 2
        if not isinstance(self.footprint, int):
            self.footprint = tuple(self.footprint)
            self.padding = tuple(self.padding)

        self.gaussian_pyramids = []
        for i, chunk_sigmas in enumerate(
            chunks(sigmas, math.ceil(len(self.sigma_list) / NUM_GPUS))
        ):
            gaussian_pyramid = nn.Conv2d(
                1,
                len(chunk_sigmas),
                2 * self.max_radius + 1,
                bias=False,
                padding=self.max_radius,
                padding_mode="zeros",
            )
            for j, s in enumerate(chunk_sigmas):
                radius = int(truncate * s + 0.5)
                kernel = torch_gaussian_kernel(width=2 * radius + 1, sigma=s.item())
                pad_size = self.max_radius - radius
                if pad_size > 0:
                    padded_kernel = nn.ConstantPad2d(pad_size, 0)(kernel)
                else:
                    padded_kernel = kernel
                gaussian_pyramid.weight.data[j].copy_(padded_kernel)
            max_pool = nn.MaxPool3d(
                kernel_size=self.footprint, padding=self.padding, stride=1
            )
            self.gaussian_pyramids.append((gaussian_pyramid, chunk_sigmas, max_pool))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lms = []
        for i, (gaussian_pyramid, sigmas, max_pool) in enumerate(
            self.gaussian_pyramids
        ):
            gaussian_images = gaussian_pyramid(
                input.to(f"cuda:{i}", non_blocking=PIN_MEMORY)
            )
            # computing difference between two successive Gaussian blurred images
            # multiplying with standard deviation provides scale invariance
            dog_images = (gaussian_images[0][:-1] - gaussian_images[0][1:]) * (
                sigmas[:-1].unsqueeze(0).unsqueeze(0).T
            )
            image_max = max_pool(dog_images.unsqueeze(0)).squeeze(0)
            mask = dog_images == image_max
            mask &= dog_images > self.threshold
            # local_maxima = mask.nonzero().cpu().numpy()

            # np.nonzero is faster than torch.nonzero()
            local_maxima = np.column_stack(mask.cpu().numpy().nonzero())
            # lm = local_maxima.astype(np.float64)

            lm = local_maxima.astype(np.float64)

            # translate final column of lm, which contains the index of the
            # sigma that produced the maximum intensity value, into the sigma
            sigmas_of_peaks = sigmas[local_maxima[:, 0]].cpu().numpy()

            # Remove sigma index and replace with sigmas
            lm = np.hstack([lm[:, 1:], sigmas_of_peaks[np.newaxis].T])
            lms.append(lm)
        lms = np.vstack(lms)
        if self.prune:
            blobs = _prune_blobs(lms, self.overlap, sigma_dim=1)
        else:
            blobs = lms

        # blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        return blobs


def torch_dog(dataloader, min_sigma=1, max_sigma=30, prune=False, overlap=0.5):
    with torch.no_grad():
        dog = DifferenceOfGaussians(
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            sigma_bins=50,
            footprint=np.array((3, 3, 3)),
            prune=prune,
            overlap=overlap,
            threshold=0.003,
        )
        for i, (gaussian_pyramid, sigmas, max_pool) in enumerate(dog.gaussian_pyramids):
            dog.gaussian_pyramids[i] = (
                gaussian_pyramid.to(f"cuda:{i}", non_blocking=PIN_MEMORY),
                sigmas.to(f"cuda:{i}", non_blocking=PIN_MEMORY),
                max_pool.to(f"cuda:{i}", non_blocking=PIN_MEMORY),
            )

        for p in dog.parameters():
            p.requires_grad = False
        dog.eval()
        for i, img_tensor in enumerate(dataloader):
            blobs = dog(img_tensor)
            yield blobs
            if i > 10:
                break


def torch_dog_test():
    with torch.no_grad():
        input = torch.rand(1, 1, 1000, 1000)
        input[0, 0, 400:600, 400:600] = 1
        input.to(DEVICE)
        make_figure(input.detach().numpy()[0][0]).show()
        dogs = torch_dog(input)
        for d in dogs:
            make_figure(d.detach().numpy()).show()


def torch_dog_img_test():
    # this is a stupid hack because running remote and profiling messes with file paths
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=10)
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=True, num_workers=4
    )

    for blobs in torch_dog(train_dataloader, prune=True):
        print("blobs: ", len(blobs))
        make_circles_fig(screenshot[0].squeeze(0).numpy(), blobs).show()
        counts, bin_centers, _ = plt.hist([r for (_, _, r) in blobs], bins=256)
        plt.show()


def main():
    plif_dataset = PLIF(plif_dir=DATA_DIR)
    plif_dataloader = torch.utils.data.DataLoader(
        plif_dataset, batch_size=1, pin_memory=True, num_workers=4
    )

    for i, blobs in enumerate(torch_dog(plif_dataloader, prune=True)):
        print("blobs: ", len(blobs))
        make_circles_fig(plif_dataset[i].squeeze(0).numpy(), blobs).show()
        counts, bin_centers, _ = plt.hist([r for (_, _, r) in blobs], bins=256)
        plt.show()


def test_gaussian_kernel():
    t = torch_gaussian_kernel()
    X = np.arange(1, 22)
    Y = np.arange(1, 22)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        X, Y, t.numpy(), cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    # torch_dog_test()
    # a = torch.zeros(10, dtype=torch.bool)
    # print(a.int())
    set_start_method("spawn")
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
    #     torch_dog_img_test()
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # torch_dog_img_test()
    main()
