import time

import torch

from nn_dog import DEVICE, PIN_MEMORY, DATA_DIR
from nn_dog.data import PLIF
# noinspection PyUnresolvedReferences
from nn_dog.model import DifferenceOfGaussiansStandardConv
# noinspection PyUnresolvedReferences
from sk_image.preprocess import make_figure


def torch_dog(dataloader, dog):
    for img_tensor in dataloader:
        img_tensor = img_tensor.to(DEVICE, non_blocking=PIN_MEMORY)
        mask, local_maxima = dog(img_tensor)
        for m, l in zip(mask, local_maxima):
            blobs = dog.make_blobs(m, l)
            yield blobs


def main():
    print(DATA_DIR)
    plif_dataset = PLIF(plif_dir=DATA_DIR)
    plif_dataloader = torch.utils.data.DataLoader(
        plif_dataset, batch_size=1, pin_memory=PIN_MEMORY, num_workers=1
    )

    img_height, img_width = plif_dataset[0].squeeze(0).numpy().shape

    with torch.no_grad():
        model = DifferenceOfGaussiansStandardConv(
            min_sigma=1,
            max_sigma=10,
            overlap=0.9,
            threshold=0.012,
            prune=False,
            sigma_bins=20,
        ).to(DEVICE, non_blocking=PIN_MEMORY)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        start = time.monotonic()
        for i, blobs in enumerate(torch_dog(plif_dataloader, model)):
            print(time.monotonic() - start)
            start = time.monotonic()


if __name__ == "__main__":
    # torch_dog_img_test()
    # set_start_method("spawn")
    main()
    # dog_train()
