import csv

import numpy as np
from PIL import Image
from glumpy import app, gloo, gl
from glumpy.ext import png
from skimage.filters import gaussian
from skimage.io import imread
from skimage.util import random_noise

# noinspection PyUnresolvedReferences
from sk_image.blob import make_circles_fig

# noinspection PyUnresolvedReferences
from sk_image.preprocess import make_figure


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


if __name__ == "__main__":
    window_size = 1000
    V = np.zeros(100, [("center", np.float32, 3), ("radius", np.float32, 1)])
    cluster_mean = np.random.uniform(window_size // 4, 3 * window_size // 4, size=2)
    cluster_std = np.diag(
        np.random.uniform((window_size // 8) ** 2, (window_size // 4) ** 2, size=2)
    )
    V["center"][:, :2] = np.round(
        np.random.multivariate_normal(mean=cluster_mean, cov=cluster_std, size=len(V))
    )
    # V["center"][:, 2] = np.random.uniform(0, 1, len(V))
    V["radius"] = np.random.uniform(5, 30, len(V))

    with open("truth.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "z", "r"])
        for ((x, y, z), r) in V:
            if (0,0) <= (x,y) <= (window_size, window_size):
                writer.writerow([x, y, z, r])

    app.use("glfw")
    window = app.Window(window_size, window_size)

    @window.event
    def on_resize(width, height):
        points["resolution"] = width, height

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)

    @window.event
    def on_draw(dt):
        window.clear()
        points.draw(gl.GL_POINTS)
        gl.glReadPixels(
            0,
            0,
            window.width,
            window.height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            framebuffer,
        )
        png.from_array(framebuffer, "RGB").save("screenshot.png")

    points = gloo.Program("vertex_shader.glsl", "fragment_shader.glsl")
    points.bind(V.view(gloo.VertexBuffer))

    framebuffer = np.zeros((window.height, 3 * window.width), dtype=np.uint8)
    app.run(framecount=1)

    im = imread("screenshot.png", as_gray=True)
    # make_figure(im).show()
    background = np.zeros_like(im)
    noisy = random_noise(background, mode="gaussian")
    noisy = random_noise(noisy, mode="poisson")

    blobs = []
    for (y, x, _z), r in V:
        blobs.append((x, y, r))
        mask = create_circular_mask(window_size, window_size, (y, x), r * 0.9)
        noisy[mask] = 0
    noise_strength = 2
    res = gaussian(2 * noisy + im, sigma=1)
    # make_circles_fig(res, np.array(blobs)).show()
    Image.fromarray((res * 255).astype(np.uint8)).save("screenshot.png")
