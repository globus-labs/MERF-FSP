import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve as convolve
from skimage.util import random_noise
import tikzplotlib


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    return kern1d / sum(kern1d)


FONT_SIZE = 20


def oned_first_order_filter():
    pth = "imgs/oned_first_order_filter"
    os.makedirs(pth, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    x = np.arange(start=1, stop=100)
    edge = np.piecewise(x, [x < 45, x >= 45], [0, 10])
    edge = edge + random_noise(edge, mode="gaussian", var=1e-1)
    filter = [1, 0, -1]
    for i in range(1, len(edge)):
        for ax in axes:
            ax.clear()
            ax.set_xlim([0, 100])
            ax.set_ylim([-1, 12])

        m_full = convolve(edge[:i], filter, mode="valid")
        axes[0].plot(edge)
        axes[0].set_title("Signal $I$", fontsize=FONT_SIZE)

        axes[1].set_ylim([-2, 2])
        axes[1].step(
            np.arange(1, 105), np.pad([1, 1, -1, -1][::-1], (i, 100 - i)), color="red"
        )
        axes[1].set_title("Filter $k = \frac{d}{dx}$", fontsize=FONT_SIZE)

        axes[2].plot(
            np.pad(m_full, (0, 100), mode="constant", constant_values=0), color="purple"
        )
        axes[2].set_title("Response $I * k$", fontsize=FONT_SIZE)

        fig.tight_layout(pad=3.0)
        fig.savefig(f"{pth}/{i:02d}.png")


def oned_smoothed_first_order_filter():
    pth = "imgs/oned_smoothed_first_order_filter"
    os.makedirs(pth, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    x = np.arange(start=1, stop=100)
    edge = np.piecewise(x, [x < 45, x >= 45], [0, 10])
    edge = edge + random_noise(edge, mode="gaussian", var=1e-1)
    smoothed_edge = convolve(gkern(), edge, mode="valid")
    filter = [1, 0, -1]

    axes[0].plot(edge)
    axes[0].set_title("Signal $I$", fontsize=FONT_SIZE)

    axes[1].plot(smoothed_edge)
    axes[1].set_title("Smoothed Signal $I * g$", fontsize=FONT_SIZE)

    m_full = convolve(smoothed_edge, filter, mode="valid")
    axes[2].plot(m_full, color="purple")
    axes[2].set_title("Response $(I * g) * k$", fontsize=FONT_SIZE)

    fig.tight_layout(pad=3.0)
    fig.savefig(f"{pth}/fin.png")


def convolution_is_associative():
    pth = "imgs/convolution_is_associative"
    os.makedirs(pth, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    x = np.arange(start=1, stop=100)
    edge = np.piecewise(x, [x < 45, x >= 45], [0, 10])
    edge = edge + random_noise(edge, mode="gaussian", var=1e-1)
    filter = [1, 0, -1]
    g_kern = gkern()
    smoothed_g_kern = convolve(g_kern, filter, mode="valid")

    for i in range(1, len(edge)):
        for ax in axes:
            ax.clear()
            ax.set_xlim([0, 100])

        axes[0].plot(edge)
        axes[0].set_title("Signal $I$", fontsize=FONT_SIZE)

        axes[1].plot(np.pad(smoothed_g_kern[::-1], (i, 100 - i)), color="red")
        axes[1].set_title(
            "Smoothed Filter $g * k = \\frac{d}{dx} g$", fontsize=FONT_SIZE
        )

        m_full = convolve(
            np.pad(edge[:i], (0, 110 - i), mode="constant", constant_values=0)[10:],
            smoothed_g_kern,
        )
        axes[2].set_ylim([-0.5, 2.5])
        axes[2].plot(m_full, color="purple")
        axes[2].set_title("Response $I * (g * k)$", fontsize=FONT_SIZE)

        fig.tight_layout(pad=3.0)
        fig.savefig(f"{pth}/{i:02d}.png")


def second_order_filter():
    pth = "imgs/second_order_filter"
    os.makedirs(pth, exist_ok=True)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
    x = np.arange(start=1, stop=100)
    edge = np.piecewise(x, [x < 45, x >= 45], [0, 10])
    edge = edge + random_noise(edge, mode="gaussian", var=1e-1)
    filter = [1, 0, -1]
    g_kern = gkern()
    smoothed_g_kern = convolve(g_kern, filter, mode="valid")
    smoothed_g_kern = convolve(smoothed_g_kern, filter, mode="valid")
    print(sum(smoothed_g_kern))

    for i in range(1, len(edge)):
        for ax in axes:
            ax.clear()
            ax.set_xlim([0, 100])

        axes[0].plot(edge)
        axes[0].set_title("Signal $I$", fontsize=FONT_SIZE)

        axes[1].plot(np.pad(smoothed_g_kern[::-1], (i, 100 - i)), color="red")
        axes[1].set_title(
            "Smoothed Second Order Filter $g * k^2 = \\frac{d^2}{dx^2} g$",
            fontsize=FONT_SIZE,
        )

        m_full = convolve(
            np.pad(edge[:i], (0, 110 - i), mode="constant", constant_values=0)[10:],
            smoothed_g_kern,
        )
        axes[2].set_ylim([-1, 1])
        axes[2].plot(m_full, color="purple")
        axes[2].set_title("Response $I * (g * k^2)$", fontsize=FONT_SIZE)

        fig.tight_layout(pad=3.0)
        fig.savefig(f"{pth}/{i:02d}.png")


def multiple_responses():
    f1, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 5))
    for i in range(4):
        x = np.arange(start=1, stop=100)
        edge = np.piecewise(
            x,
            [
                x < 45 - (3 - i) * 5,
                np.logical_and(x >= 45 - (3 - i) * 5, x < 55 + (3 - i) * 5),
                x >= 55 + (3 - i) * 5,
            ],
            [0, 10, 0],
        )
        edge = edge + random_noise(edge, mode="gaussian", var=1e-1)
        filter = [1, 0, -1]
        g_kern_filter = convolve(
            convolve(gkern(), filter, mode="valid"), filter, mode="valid"
        )
        g_kern_filter = g_kern_filter[::-1]
        padded_gkern = np.pad(
            g_kern_filter, pad_width=(30, 30), mode="constant", constant_values=0
        )

        smoothed_edge = gaussian_filter1d(edge, sigma=2)
        m_full = convolve(smoothed_edge, g_kern_filter, mode="valid")

        ### Plot the input, repsonse function, and analytic result
        axes.ravel()[i].plot(edge)
        axes.ravel()[i].set_title("Signal")

        # ax2.plot(np.arange(-38, 39), padded_gkern, color="red")
        # ax2.set_title(
        #     "$g * k * k =\\frac{d^2}{dx^2} g$"
        # )
        # ax2.set_xticks([-1,0,1])

        axes.ravel()[i + 4].plot(m_full, color="purple")
        axes.ravel()[i + 4].set_title("Filtered signal")
    plt.show()


def maxes():
    a = [
        0.234704852104187,
        0.40904472811147574,
        0.3927840051800013,
        0.3203098058607429,
        0.25212803117930893,
        0.19899935787543654,
        0.15924187002703546,
        0.12950117417611182,
        0.10697871189564467,
        0.08964827636256814,
        0.07609164302702993,
        0.06531082844361663,
        0.05659240918699652,
        0.0494045759330038,
        0.04332736701704562,
        0.038032823975663634,
        0.03326337417587638,
        0.02882724844617769,
        0.024604673370486125,
        0.02052367448981386,
        0.016564721846953036,
        0.012741939013358206,
        0.009088923438685016,
        0.005651089787133969,
        0.0024732487881556154,
        -0.0004054924356751144,
        -0.002953998630400747,
        -0.005153494936530479,
        -0.006998500414192677,
        -0.008488055605557748,
        -0.009636013949057087,
        -0.010460863907937892,
        -0.010986547116190196,
        -0.011237978647695853,
        -0.011246873572235926,
        -0.011043115117354318,
        -0.01065294157480821,
        -0.01010930004878901,
        -0.00943600871716626,
        -0.008658032395178452,
    ]
    b = [
        0.06421422958374023,
        0.24336371865123513,
        0.4262143554538489,
        0.5205073740333319,
        0.5296450729668141,
        0.4934152498841286,
        0.44119364123791455,
        0.38759921883232895,
        0.3384895500540734,
        0.29558950051665306,
        0.25887893624603747,
        0.2276818323042244,
        0.20119336202740667,
        0.1785679944511503,
        0.1590503386873752,
        0.14201373634859918,
        0.1269415755569935,
        0.11343061229446902,
        0.10120459200814366,
        0.09007429744815454,
        0.07991569200530649,
        0.07065782571677119,
        0.06227029497735203,
        0.05472172974376008,
        0.04798468904569745,
        0.04203927580965683,
        0.03684906341135502,
        0.032366278083063665,
        0.028547014910727742,
        0.02534150196937844,
        0.02268522826489061,
        0.02052440554718487,
        0.01879931136965752,
        0.017461180448299272,
        0.016449633731972425,
        0.015720457764109595,
        0.01522613319568336,
        0.014933664330746978,
        0.014802630003541708,
        0.014799675437388943,
    ]
    c = [
        0.0021944642066955566,
        0.033044022321701054,
        0.10931006364524365,
        0.21562477722764015,
        0.33111935734748843,
        0.4349342290312052,
        0.5126379159837962,
        0.5597608900815249,
        0.5796057391166688,
        0.5786696621403098,
        0.5635962462052703,
        0.5398198548518122,
        0.5113381634652614,
        0.4808178344555199,
        0.4499993274733424,
        0.4199565195478499,
        0.391237385571003,
        0.36416515488177537,
        0.33885292006656526,
        0.31531168016605077,
        0.2935059770941734,
        0.27335511556826536,
        0.2547917036339641,
        0.23772226167842744,
        0.2220457999035716,
        0.20767640182748437,
        0.19452191304415464,
        0.18248654704075307,
        0.17149176375940442,
        0.1614234766131267,
        0.15223541883751748,
        0.1438187475549057,
        0.13611421883106234,
        0.12905267106834797,
        0.12257051276043057,
        0.116608411911875,
        0.11111703688278794,
        0.10605523204430937,
        0.10137016657739878,
        0.09703307546442375,
    ]
    plt.plot(a, label="a")
    plt.plot(b, label="b")
    plt.plot(c, label="c")
    plt.xlabel("scale")
    plt.legend()
    plt.yticks([])
    # plt.show()
    tikzplotlib.save("scalespace.tex")
    # for i in range(len(a)):
    #     plt.clf()
    #     plt.plot(a)
    #     plt.axvline(x=i, linestyle="--", color="black")
    #     plt.savefig(f"/Users/maksim/dev_projects/merf/nns/dog/dogs/a{i:02d}.png")
    # for i in range(len(b)):
    #     plt.clf()
    #     plt.plot(b)
    #     plt.axvline(x=i, linestyle="--", color="black")
    #     plt.savefig(f"/Users/maksim/dev_projects/merf/nns/dog/dogs/b{i:02d}.png")
    # for i in range(len(c)):
    #     plt.clf()
    #     plt.plot(c)
    #     plt.axvline(x=i, linestyle="--", color="black")
    #     plt.savefig(f"/Users/maksim/dev_projects/merf/nns/dog/dogs/c{i:02d}.png")



if __name__ == "__main__":
    maxes()
    # oned_first_order_filter()
    # oned_smoothed_first_order_filter()
    # convolution_is_associative()
    # second_order_filter()
    # multiple_responses()