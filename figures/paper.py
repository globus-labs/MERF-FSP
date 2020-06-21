import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib.lines import Line2D
from scipy.stats import stats
from skimage import io

from sk_image.blob import make_circles_fig


def uf_plots():
    n_lines = 30
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu2 = pd.read_csv("/Users/maksim/dev_projects/merf/figures/time_results/gpu_model_parallel_fft_run_times_uf_2_gpus.csv")
    gpu3 = pd.read_csv("/Users/maksim/dev_projects/merf/figures/time_results/gpu_model_parallel_fft_run_times_uf_3_gpus.csv")
    gpu4 = pd.read_csv("/Users/maksim/dev_projects/merf/figures/time_results/gpu_model_parallel_fft_run_times_uf_4_gpus.csv")
    for i in range(3, 31):
        gpu22 = gpu2[gpu2["max_sigma"] == i]["time"].values
        gpu33 = gpu3[gpu3["max_sigma"] == i]["time"].values
        gpu44 = gpu4[gpu4["max_sigma"] == i]["time"].values

        ax.plot(np.arange(3, 41), gpu22[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 41), gpu33[1:], "-.", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 41), gpu44[1:], ":", c=cmap.to_rgba(i + 1))


    ax.set_xticks([0, 3, 10, 20, 30, 40])
    ax.set_yscale('log')
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    gpu2line = Line2D([0], [0], color="red", linewidth=2, linestyle="--")
    gpu3line = Line2D([0], [0], color="red", linewidth=2, linestyle="-.")
    gpu4line = Line2D([0], [0], color="red", linewidth=2, linestyle=":")
    fig.legend(
        [gpu2line, gpu3line, gpu4line],
        ["2", "3", "4"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    tikzplotlib.save("multiple_gpu.tex")

def two_plots(fp1, plot_name1, fp2, plot_name2, tikz=False):
    n_lines = 30
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv(fp1)
    gpu_standard_df = pd.DataFrame(
        gpu_standard_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )
    gpu_df = pd.read_csv(fp2)
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(3, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    ax.set_yscale('log')
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        [plot_name1, plot_name2],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    if tikz:
        tikzplotlib.save(f"{plot_name1}_{plot_name2}.tex", figure=fig)
    else:
        fig.show()

def gpu_gpu_parallel():
    n_lines = 30
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("time_results/gpu_model_parallel_fft_run_times_uf_4_gpus.csv")
    gpu_standard_df = pd.DataFrame(
        gpu_standard_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )
    gpu_df = pd.read_csv("time_results/gpu_fft_run_times_uf.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(3, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    ax.set_yscale('log')
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        ["parallel", "single"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()

def gpu_gpu_copy():
    n_lines = 30
    c = np.arange(1, n_lines + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_standard_df = pd.DataFrame(
        gpu_standard_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        ["with copy", "no copy"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def gpu_gpu():
    n_lines = 30
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    gpu_standard_df = pd.read_csv("gpu_standardconv_run_times.csv")
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    for i in range(2, 31):
        gpu_st = gpu_standard_df[gpu_standard_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(gpu_st) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(3, 51), gpu_st[1:], "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(3, 51), gpu[1:], c=cmap.to_rgba(i + 1))

    ax.set_yscale("log")
    ax.set_title("GPU-FFTConv vs. GPU-StandardConv performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    fig.legend(
        [cpuline, gpuline],
        ["StandardConv", "FFTConv"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def cpu_gpu():
    n_lines = 29
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    cpu_df = pd.read_csv("cpu_run_times.csv")
    cpu_df = pd.DataFrame(cpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_df = pd.read_csv("gpu_fftconv_run_times.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_copy_df = pd.read_csv("gpu_fftconv_with_img_copy_run_times.csv")
    gpu_copy_df = pd.DataFrame(
        gpu_copy_df.groupby(["n_bin", "max_sigma"]).mean().to_records()
    )

    for i in range(2, 31):
        # cpu = cpu_df[cpu_df['max_sigma'] == i]['time'].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        gpu_copy = gpu_copy_df[gpu_copy_df["max_sigma"] == i]["time"].values
        # if len(cpu) == 0 or len(gpu) == 0: continue

        # ax.plot(np.arange(2,51), cpu,"--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu * 1000, c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu_copy * 1000, ":", c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_title("GPU vs. CPU performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    gpu_copy_line = Line2D([0], [0], color="blue", linewidth=2, linestyle=":")
    fig.legend(
        [cpuline, gpuline, gpu_copy_line],
        ["cpu", "gpu", "gpu_copy"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    fig.show()
    # tikzplotlib.save("test.tex", figure=fig)


def cpu_gpu_standard():
    n_lines = 29
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)
    cmap.set_array([])

    fig, ax = plt.subplots(dpi=100)

    cpu_df = pd.read_csv("cpu_run_times.csv")
    cpu_df = pd.DataFrame(cpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())
    gpu_df = pd.read_csv("gpu_standard_run_times_with_img_copy.csv")
    gpu_df = pd.DataFrame(gpu_df.groupby(["n_bin", "max_sigma"]).mean().to_records())

    for i in range(3, 31):
        cpu = cpu_df[cpu_df["max_sigma"] == i]["time"].values
        gpu = gpu_df[gpu_df["max_sigma"] == i]["time"].values
        if len(cpu) == 0 or len(gpu) == 0:
            continue

        ax.plot(np.arange(2, 51), cpu, "--", c=cmap.to_rgba(i + 1))
        ax.plot(np.arange(2, 51), gpu, c=cmap.to_rgba(i + 1))

    # ax.set_yscale('log')
    ax.set_title("GPU vs. CPU performance")
    ax.set_xticks([0, 3, 10, 20, 30, 40, 50])
    ax.set_ylabel("time (s)")
    ax.set_xlabel("filters")
    fig.colorbar(cmap, ticks=c, label="max radius")

    cpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="--")
    gpuline = Line2D([0], [0], color="blue", linewidth=2, linestyle="-")
    gpu_copy_line = Line2D([0], [0], color="blue", linewidth=2, linestyle=":")
    fig.legend(
        [cpuline, gpuline, gpu_copy_line],
        ["cpu", "gpu"],
        loc="upper left",
        bbox_to_anchor=(0.1, 0.9),
    )
    # fig.show()
    tikzplotlib.save("test.tex", figure=fig)


# https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
def intersection_area(d, R, r):
    if d <= abs(R - r):
        # One circle is entirely enclosed in the other.
        return np.pi * min(R, r) ** 2
    if d >= r + R:
        # The circles don't overlap at all.
        return 0

    r2, R2, d2 = r ** 2, R ** 2, d ** 2
    alpha = np.arccos((d2 + r2 - R2) / (2 * d * r))
    beta = np.arccos((d2 + R2 - r2) / (2 * d * R))
    return (
            r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))
    )


def circle_iou(c1, c2):
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    inters = intersection_area(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), r1, r2)
    unio = (4 / 3) * np.pi * (r1 ** 3 + r2 ** 3)
    return inters / unio


# for each detection that has a confidence score > threshold:
#
#     among
#     the
#     ground - truths, choose
#     one
#     that
#     belongs
#     to
#     the
#     same
#     class and has the highest IoU with the detection
#
#
#     if no ground-truth can be chosen or IoU < threshold (e.g., 0.5):
#         the
#         detection is a
#         false
#         positive
#     else:
#         the
#         detection is a
#         true
#         positive


def accuracy(save=True):
    cpu_precisions = []
    cpu_recalls = []
    gpu_precisions = []
    gpu_recalls = []
    for i in range(1, 100 + 1):
        print(i)
        truth_fp = f"../simulation/test_data/truth{i}.csv"
        cpu_res = f"accuracy_results/cpu/screenshot{i}.png.res"
        gpu_res = f"accuracy_results/gpu/screenshot{i}.png.res"

        truth_csv = pd.read_csv(truth_fp)
        # make_circles_fig(img, truth_csv[['y', 'x', 'r']].to_numpy()).show()

        cpu_res = np.loadtxt(cpu_res)
        # make_circles_fig(img, cpu_res).show()
        fps = 0
        tps = 0
        for (x, y, r) in cpu_res:
            ious = truth_csv.apply(lambda row: circle_iou((x, y, r), (row['y'], row['x'], row['r'])), axis=1)
            ious = ious[ious > 0].sort_values()
            if len(ious):
                tps += 1
                fps += len(ious[1:])

        precision = tps / len(cpu_res)
        recall = fps / len(truth_csv)
        cpu_precisions.append(precision)
        cpu_recalls.append(recall)

        gpu_res = np.loadtxt(gpu_res)
        # make_circles_fig(img, gpu_res).show()
        fps = 0
        tps = 0
        for (x, y, r) in gpu_res:
            ious = truth_csv.apply(lambda row: circle_iou((x, y, r), (row['y'], row['x'], row['r'])), axis=1)
            ious = ious[ious > 0].sort_values()
            if len(ious):
                tps += 1
                fps += len(ious[1:])

        precision = tps / len(gpu_res)
        recall = fps / len(truth_csv)
        gpu_precisions.append(precision)
        gpu_recalls.append(recall)

    df = pd.DataFrame(data={
        "cpu_precision": cpu_precisions,
        "cpu_recall": cpu_recalls,
        "gpu_precision": gpu_precisions,
        "gpu_recall": gpu_recalls,
    })

    if save:
        df.to_csv("precision_recall.csv")
    else:
        print(df)

def accuracy_pdf():
    df = pd.read_csv("/Users/maksim/dev_projects/merf/figures/accuracy_results/precision_recall.csv")
    prec = df['cpu_precision'] - df['gpu_precision']
    rec = df['cpu_recall'] - df['gpu_recall']

    print(stats.describe(prec))
    print(stats.describe(rec))

    plt.hist(prec, bins=20, label="$CPU_p - GPU_p$", alpha=.5)
    plt.hist(rec, bins= 100, label="$CPU_r - GPU_r$", alpha=.5)
    plt.legend()
    tikzplotlib.save("accuracy.tex")
    # plt.show()

def accuracy_plot():
    phi = np.linspace(0, 2 * np.pi, 100)
    x = np.sin(phi)
    y = np.cos(phi)
    rgb_cycle = np.vstack((  # Three sinusoids
        .5 * (1. + np.cos(phi)),  # scaled to [0,1]
        .5 * (1. + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
        .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T  # Shape = (60,3)

    df = pd.read_csv("/Users/maksim/dev_projects/merf/figures/accuracy_results/precision_recall.csv")
    for i in range(100):
        plt.scatter(df['cpu_recall'].values[i], df['cpu_precision'].values[i], c=rgb_cycle[i], s=90, marker="+")
        plt.scatter(df['gpu_recall'].values[i], df['gpu_precision'].values[i], c=rgb_cycle[i], s=90, marker="x")
    plt.xlabel("recall")
    plt.ylabel("precision")
    # plt.legend()
    # plt.show()
    tikzplotlib.save("test.tex")

    df = pd.read_csv("figures/test.tex", sep=" ")
    splits = open("figures/test.tex").read().split("%")
    rows = []
    for i in range(20):
        rows.append(splits[2 * i])
        rows.append("\nx y\n")
        rows.append(f"{df['x'][i]} {df['y'][i]}\n")
        rows.append(splits[2 * i + 1])
        rows.append("\nx y\n")
        rows.append(f"{df['x'][i + 100]} {df['y'][i + 100]}\n")

    print("".join(rows), file=open("test2.tex", "w"))

if __name__ == "__main__":
    # gpu_gpu()
    # gpu_gpu_copy()
    # cpu_gpu_copy()
    # cpu_gpu()
    # cpu_gpu_standard()
    # accuracy(save=True)
    # accuracy_plot()
    accuracy_pdf()
    # gpu_gpu_parallel()
    # two_plots(
    #    "/Users/maksim/dev_projects/merf/figures/time_results/gpu_standard_run_times_uf.csv",
    #    "standard",
    #    "/Users/maksim/dev_projects/merf/figures/time_results/gpu_fft_run_times_uf.csv",
    #    "fft"
    # )

    # two_plots(
    #     "/Users/maksim/dev_projects/merf/figures/time_results/gpu_fftconv_run_times.csv",
    #     "joe",
    #     "/Users/maksim/dev_projects/merf/figures/time_results/gpu_fft_run_times_uf.csv",
    #     "uf"
    # )
    # two_plots(
    #     "/Users/maksim/dev_projects/merf/figures/time_results/gpu_model_parallel_fft_run_times_uf_4_gpus.csv",
    #     "4 gpus",
    #     "/Users/maksim/dev_projects/merf/figures/time_results/gpu_model_parallel_fft_run_times_uf_2_gpus.csv",
    #     "2 gpus"
    # )
    # uf_plots()
