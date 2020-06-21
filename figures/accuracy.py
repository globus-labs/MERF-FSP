import glob
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from figures.run_times import min_sigma, mx_sigma, max_bin
from nn_dog import PIN_MEMORY, DEVICE
from nn_dog.data import SimulPLIF, PLIF
from nn_dog.model import DifferenceOfGaussiansFFT
from sk_image.blob import cpu_blob_dog


def cpu_accuracy():
    glob_str = (
        str(
            Path(os.path.dirname(os.path.realpath(__file__)))
            / Path("../simulation/test_data/")
        )
        + "/*.png"
    )
    print(glob_str)
    for image_pth in sorted(glob.glob(glob_str)):
        screenshot = SimulPLIF(img_path=image_pth, num_repeats=1, load_truth=False)
        blobs = cpu_blob_dog(
            screenshot[0],
            min_sigma=min_sigma,
            max_sigma=mx_sigma,
            overlap=0.5,
            threshold=0.1,
            sigma_bins=max_bin,
            prune=False,
        )
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        # make_circles_fig(screenshot[0].numpy(), blobs).show()
        # break
        fn = str(
            Path(os.path.dirname(os.path.realpath(__file__)))
            / f"accuracy_results/cpu/{os.path.basename(image_pth)}.res"
        )
        np.savetxt(fn, blobs)
        print(fn)


def gpu_accuracy():
    test_img_dir = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/test_data/"
    )

    screenshot = PLIF(plif_dir=test_img_dir, ext="png", include_paths=True)
    img_height, img_width = screenshot[0][0].squeeze(0).numpy().shape

    train_dataloader = DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY, shuffle=False
    )

    with torch.no_grad():
        model = DifferenceOfGaussiansFFT(
            img_height=img_height,
            img_width=img_width,
            min_sigma=min_sigma,
            max_sigma=mx_sigma,
            overlap=0.5,
            threshold=0.1,
            prune=False,
            sigma_bins=max_bin,
        ).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

    for img_tensor, image_pth in train_dataloader:
        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        mask, local_maxima = model(img_tensor)
        m, l = next(zip(mask, local_maxima))
        blobs = model.make_blobs(m, l)
        # make_circles_fig(screenshot[0][0].numpy(), blobs).show()
        # break
        image_pth = list(image_pth)[0]
        fn = str(
            Path(os.path.dirname(os.path.realpath(__file__)))
            / f"accuracy_results/gpu/{os.path.basename(image_pth)}.res"
        )
        np.savetxt(fn, blobs)
        print(fn)


if __name__ == "__main__":
    # cpu_accuracy()
    gpu_accuracy()
