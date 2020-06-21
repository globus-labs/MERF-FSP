import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn
cudnn.deterministic = False
from torch.utils.data import DataLoader
from skimage import io, img_as_float
from skimage.filters import gaussian
import torch.multiprocessing as mp
from nn_dog import PIN_MEMORY, DEVICE, NUM_GPUS
from nn_dog.data import SimulPLIF
from nn_dog.model import DifferenceOfGaussiansFFT, DifferenceOfGaussiansStandardConv, DifferenceOfGaussiansFFTParallel, \
    close_pool
from sk_image.blob import cpu_blob_dog
from sk_image.enhance_contrast import stretch_composite_histogram

min_bin = 2
max_bin = 40
min_sigma = 1
mx_sigma = 30
REPEATS = 1


def cpu_run_with_copy_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    with open("cpu_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        for i in range(REPEATS):

            for max_sigma in range(min_sigma, mx_sigma + 1):
                for n_bin in range(min_bin, max_bin + 1):
                    START = time.monotonic()
                    cpu_blob_dog(
                        screenshot[0],
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        sigma_bins=n_bin,
                        prune=False,
                    )
                    res = [n_bin, max_sigma, time.monotonic() - START]
                    writer.writerow(res)
                    print(res)
                #     break
                # break


def cpu_run_times(fn):
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    with open(f"{fn}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        for i in range(REPEATS):

            for max_sigma in range(min_sigma, mx_sigma + 1):
                for n_bin in range(min_bin, max_bin + 1):
                    START = time.monotonic()
                    cpu_blob_dog(
                        screenshot[0],
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        sigma_bins=n_bin,
                        prune=False,
                    )
                    res = [n_bin, max_sigma, time.monotonic() - START]
                    writer.writerow(res)
                    print(res)
                #     break
                # break


def gpu_fft_run_times(fn):
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = DataLoader(screenshot, batch_size=1, pin_memory=PIN_MEMORY)

    with open(f"{fn}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for i in range(REPEATS):
            for max_sigma in range(min_sigma + 1, mx_sigma + 1):
                for n_bin in range(min_bin, max_bin + 1):
                    with torch.no_grad():
                        model = DifferenceOfGaussiansFFT(
                            img_height=img_height,
                            img_width=img_width,
                            min_sigma=min_sigma,
                            max_sigma=max_sigma,
                            overlap=0.9,
                            threshold=0.012,
                            prune=False,
                            sigma_bins=n_bin,
                        ).to(DEVICE, non_blocking=PIN_MEMORY)
                        for p in model.parameters():
                            p.requires_grad = False
                        model.eval()
                        torch.cuda.synchronize(DEVICE)

                        img_tensor = next(iter(train_dataloader))
                        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                        torch.cuda.synchronize(DEVICE)

                        START = time.monotonic()

                        mask, local_maxima = model(img_tensor)
                        m, l = next(zip(mask, local_maxima))
                        blobs = model.make_blobs(m, l)
                        torch.cuda.synchronize(DEVICE)

                        END = time.monotonic()

                        res = [n_bin, max_sigma, END - START]
                        writer.writerow(res)
                        print(res)
                #     break
                # break

def gpu_model_parallel_fft_run_times(fn):
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = DataLoader(screenshot, batch_size=1, pin_memory=PIN_MEMORY)

    with open(f"{fn}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansFFTParallel(
                        img_height=img_height,
                        img_width=img_width,
                        num_gpus=NUM_GPUS,
                        pin_memory=PIN_MEMORY,
                        master_device=DEVICE,
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    model.move_kernels_to_gpu()
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()

                    pool = mp.Pool(len(model.f_gaussian_pyramids), maxtasksperchild=10)
                    for i in range(REPEATS):

                        img_tensor = next(iter(train_dataloader))
                        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                        torch.cuda.synchronize(DEVICE)

                        START = time.monotonic()

                        mask, local_maxima = model(img_tensor, pool)
                        m, l = next(zip(mask, local_maxima))
                        blobs = model.make_blobs(m, l)
                        torch.cuda.synchronize(DEVICE)

                        END = time.monotonic()

                        res = [n_bin, max_sigma, END - START]
                        writer.writerow(res)
                        print(res)
                    # close_pool(pool)



def gpu_fft_run_times_with_img_copy():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)
    img_height, img_width = screenshot[0].squeeze(0).numpy().shape

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_fftconv_with_img_copy_run_times.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for i in range(REPEATS):
            for max_sigma in range(min_sigma + 1, mx_sigma + 1):
                for n_bin in range(min_bin, max_bin + 1):
                    with torch.no_grad():
                        model = DifferenceOfGaussiansFFT(
                            img_height=img_height,
                            img_width=img_width,
                            min_sigma=min_sigma,
                            max_sigma=max_sigma,
                            overlap=0.9,
                            threshold=0.012,
                            prune=False,
                            sigma_bins=n_bin,
                        ).to(DEVICE, non_blocking=PIN_MEMORY)
                        for p in model.parameters():
                            p.requires_grad = False
                        model.eval()
                        torch.cuda.synchronize(DEVICE)

                        START = time.monotonic()

                        img_tensor = next(iter(train_dataloader))
                        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                        torch.cuda.synchronize(DEVICE)

                        mask, local_maxima = model(img_tensor)
                        m, l = next(zip(mask, local_maxima))
                        blobs = model.make_blobs(m, l)
                        torch.cuda.synchronize(DEVICE)

                        END = time.monotonic()

                        res = [n_bin, max_sigma, END - START]
                        writer.writerow(res)
                        print(res)
                #     break
                # break


def gpu_standard_run_times(fn):
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1000000)
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open(f"{fn}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])

        for max_sigma in range(min_sigma + 1, mx_sigma + 1):
            for n_bin in range(min_bin, max_bin + 1):
                with torch.no_grad():
                    model = DifferenceOfGaussiansStandardConv(
                        min_sigma=min_sigma,
                        max_sigma=max_sigma,
                        overlap=0.9,
                        threshold=0.012,
                        prune=False,
                        sigma_bins=n_bin,
                    ).to(DEVICE, non_blocking=PIN_MEMORY)
                    for p in model.parameters():
                        p.requires_grad = False
                    model.eval()

                    for i in range(REPEATS):

                        img_tensor = next(iter(train_dataloader))
                        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                        torch.cuda.synchronize(DEVICE)

                        START = time.monotonic()

                        mask, local_maxima = model(img_tensor.unsqueeze(0))
                        blobs = model.make_blobs(mask, local_maxima)
                        torch.cuda.synchronize(DEVICE)

                        END = time.monotonic()

                        res = [n_bin, max_sigma, END - START]
                        writer.writerow(res)
                        print(res)
                    del model
                #     break
                # break


def gpu_standard_run_times_with_img_copy():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    screenshot = SimulPLIF(img_path=image_pth, num_repeats=1)

    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )

    with open("gpu_standard_run_times_with_img_copy.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["n_bin", "max_sigma", "time"])
        for i in range(REPEATS):

            for max_sigma in range(min_sigma + 1, mx_sigma + 1):
                for n_bin in range(min_bin, max_bin + 1):
                    with torch.no_grad():
                        model = DifferenceOfGaussiansStandardConv(
                            min_sigma=min_sigma,
                            max_sigma=max_sigma,
                            overlap=0.9,
                            threshold=0.012,
                            prune=False,
                            sigma_bins=n_bin,
                        ).to(DEVICE, non_blocking=PIN_MEMORY)
                        for p in model.parameters():
                            p.requires_grad = False
                        model.eval()

                        START = time.monotonic()

                        img_tensor = next(iter(train_dataloader))
                        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
                        torch.cuda.synchronize(DEVICE)

                        mask, local_maxima = model(img_tensor.unsqueeze(0))
                        blobs = model.make_blobs(mask, local_maxima)
                        torch.cuda.synchronize(DEVICE)

                        END = time.monotonic()

                        res = [n_bin, max_sigma, END - START]
                        writer.writerow(res)
                        print(res)
                #     break
                # break


def copy_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )

    screenshot = SimulPLIF(
        img_path=image_pth, num_repeats=100000, apply_transforms=False, load_truth=False
    )

    PIN_MEMORY = True
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )
    times = []
    img_tensor_cpu = next(iter(train_dataloader))
    for i in range(REPEATS * 10):
        START = time.monotonic()

        img_tensor = img_tensor_cpu.to(DEVICE, non_blocking=PIN_MEMORY)
        torch.cuda.synchronize(DEVICE)

        END = time.monotonic()
        times.append(END - START)
    print(f"pin memory: {PIN_MEMORY} average time:", np.mean(times[1:]))

    PIN_MEMORY = False
    train_dataloader = torch.utils.data.DataLoader(
        screenshot, batch_size=1, pin_memory=PIN_MEMORY
    )
    times = []
    img_tensor_cpu = next(iter(train_dataloader))

    for i in range(REPEATS * 10):
        START = time.monotonic()

        img_tensor = img_tensor_cpu.to(DEVICE, non_blocking=PIN_MEMORY)
        torch.cuda.synchronize(DEVICE)

        END = time.monotonic()
        times.append(END - START)
    print(f"pin memory: {PIN_MEMORY} average time:", np.mean(times[1:]))


def io_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )

    times = []
    for i in range(REPEATS):
        START = time.monotonic()
        screenshot = SimulPLIF(img_path=image_pth, num_repeats=100000, load_truth=False)
        END = time.monotonic()
        times.append(END - START)
    print(f"disk read times {np.mean(times)}")

    times = []
    for i in range(REPEATS):
        START = time.monotonic()
        screenshot = SimulPLIF(img_path=image_pth, num_repeats=100000, load_truth=False)
        screenshot[0]
        END = time.monotonic()
        times.append(END - START)
    print(f"disk read+transform times {np.mean(times)}")

    times = []
    screenshot = SimulPLIF(
        img_path=image_pth, num_repeats=100000, apply_transforms=True, load_truth=False
    )
    for i in range(REPEATS):
        START = time.monotonic()
        screenshot[0]
        END = time.monotonic()
        times.append(END - START)
    print(f"transform times {np.mean(times)}")

    times = []
    screenshot = SimulPLIF(
        img_path=image_pth, num_repeats=100000, apply_transforms=False, load_truth=False
    )
    for i in range(REPEATS):
        START = time.monotonic()
        screenshot[0]
        END = time.monotonic()
        times.append(END - START)
    print(f"sans transform sans truth times {np.mean(times)}")


def preprocess_times():
    image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
        "../simulation/screenshot.png"
    )
    # image_pth = Path(os.path.dirname(os.path.realpath(__file__))) / Path(
    #     "../data/RawData/R109_60deg_6-8-25_OHP-LS000252.T000.D000.P000.H000.PLIF1.TIF"
    # )
    img_cpu = io.imread(image_pth)

    stretch_times = []
    gaussian_times = []
    img_as_float_times = []
    torch_times = []
    for i in range(REPEATS):
        START = time.monotonic()
        img = img_as_float(img_cpu)
        END = time.monotonic()
        img_as_float_times.append(END - START)

        START = time.monotonic()
        img = stretch_composite_histogram(img)
        END = time.monotonic()
        stretch_times.append(END - START)

        START = time.monotonic()
        img = gaussian(img, sigma=1)
        END = time.monotonic()
        gaussian_times.append(END - START)

        START = time.monotonic()
        img = torch.from_numpy(img).float()
        END = time.monotonic()
        torch_times.append(END - START)

    print(f"img_as_float times {np.mean(img_as_float_times)}")
    print(f"stretch times {np.mean(stretch_times)}")
    print(f"gaussian times {np.mean(gaussian_times)}")
    print(f"torch times {np.mean(torch_times)}")

    print()

    stretch_times = []
    gaussian_times = []
    img_as_float_times = []
    torch_times = []
    for i in range(REPEATS):
        START = time.monotonic()
        img = stretch_composite_histogram(img_cpu)
        END = time.monotonic()
        stretch_times.append(END - START)

        START = time.monotonic()
        img = img_as_float(img)
        END = time.monotonic()
        img_as_float_times.append(END - START)

        START = time.monotonic()
        img = gaussian(img, sigma=1)
        END = time.monotonic()
        gaussian_times.append(END - START)

        START = time.monotonic()
        img = torch.from_numpy(img).float()
        END = time.monotonic()
        torch_times.append(END - START)

    print(f"stretch times {np.mean(stretch_times)}")
    print(f"gaussian times {np.mean(gaussian_times)}")
    print(f"img_as_float times {np.mean(img_as_float_times)}")
    print(f"torch times {np.mean(torch_times)}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # print("gpu_fft_run_times")
    # gpu_fft_run_times("gpu_fft_run_times_joe_deterministic")
    print("gpu_standard_run_times")
    gpu_standard_run_times("gpu_standard_run_times_joe_deterministic")
    #
    # print("gpu_standard_run_times_with_img_copy")
    # gpu_standard_run_times_with_img_copy()
    # print("gpu_fft_run_times_with_img_copy")
    # gpu_fft_run_times_with_img_copy()
    #
    # print("cpu_run_times")
    # cpu_run_times("cpu_run_times_uf")
    #
    # print("copy times")
    # copy_times()
    # print()
    #
    # print("io times")
    # io_times()
    # print()
    #
    # print("preprocess times")
    # preprocess_times()
    for NUM_GPUS in range(2, 5):
        print("gpu_model_parallel_fft_run_times")
        gpu_model_parallel_fft_run_times(f"gpu_model_parallel_fft_run_times_uf_{NUM_GPUS}_gpus_100_bins")
