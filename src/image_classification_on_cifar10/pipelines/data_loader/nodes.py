"""
This is a boilerplate pipeline 'data_loader'
generated using Kedro 0.18.6
"""
import logging
import ssl

import icecream
import torchvision
from icecream import ic

# ssl._create_default_https_context = ssl._create_unverified_context  # FIXME: ???


logger = logging.getLogger(__name__)


def load_transforms(train_transform_params: dict[str]):
    # TODO:
    if "train" in train_transform_params:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.RandomCrop(
                    32, padding=4, padding_mode="reflect"
                ),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )


def download_data(
    train: bool, transform_params: dict
) -> torchvision.datasets.cifar.CIFAR10:

    logger.debug("Hello")
    logger.debug(transform_params)

    return torchvision.datasets.CIFAR10(
        root="./data/01_raw/",
        train=train,
        download=True,
        transform=load_transforms(transform_params),
    )
