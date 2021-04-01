import csv
import os
import time
from pathlib import Path

import torch

from nn_dog.data import Trivial
from nn_dog.main import torch_dog, PIN_MEMORY
from sk_image.blob import cpu_blob_dog

min_bin = 2
max_bin = 51
min_sigma = 1
mx_sigma = 31


def cpu_run_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=100)
    # make_figure(screenshot[0][0].squeeze(0).numpy()).show()
    with open("cpu_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        for max_sigma in range(min_sigma + 1, mx_sigma):
            for n_bin in range(min_bin, max_bin + 1):
                start = time.monotonic()
                cpu_blob_dog(
                    screenshot[0],
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    overlap=0.9,
                    threshold=0.012,
                    sigma_bins=n_bin,
                    prune=False,
                )
                # n_bin = int(np.mean(np.log(n_bin / 1) / np.log(1.6) + 1))
                res = [n_bin, max_sigma, time.monotonic() - start]
                writer.writerow(res)
                print(res)


def gpu_run_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = Trivial(img_path=image_pth, num_repeats=100)
    # make_figure(screenshot[0][0].squeeze(0).numpy()).show()
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        start = time.monotonic()
        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                for i, blobs in enumerate(
                    torch_dog(
                        train_dataloader,
                        img_height=img_height,
                        img_width=img_width,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    )
                ):
                    res = [n_bin, max_sigma, time.monotonic() - start]
                    writer.writerow(res)
                    print(res)
                    start = time.monotonic()
                    break


if __name__ == "__main__":
    gpu_run_times()
    # cpu_run_times()
