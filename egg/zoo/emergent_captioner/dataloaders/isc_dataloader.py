# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
from torchvision.datasets import VisionDataset

from egg.zoo.emergent_captioner.dataloaders.utils import get_transform


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ISC2021Dataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = "/homedtcl/nrakotonirina/datasets/isc2021",
        test_file: str = "simimage_dataset.txt",
        num_distractors: int = 63,
        transform: Optional[Callable] = None,
    ):
        super(ISC2021Dataset, self).__init__(dataset_dir, transform=transform)
        dataset_dir = Path(dataset_dir)

        with open(dataset_dir / test_file) as f:
            images_list = set(f.read().splitlines())
        self.samples = []
        num_distractors = min(num_distractors, 63)
        for sample in images_list:
            curr_sample = sample.split("\t")
            query_image = dataset_dir / "images" / "queries" / f"{curr_sample[0]}.jpg"
            reference_images = [
                dataset_dir / "images" / "references" / f"{i}.jpg"
                for i in curr_sample[1 : num_distractors + 2]
            ]
            self.samples.append([query_image] + reference_images)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        images_path = self.samples[index]
        images = [pil_loader(i) for i in images_path]
        if self.transform:
            images = [self.transform(i) for i in images]
        return images[0], images[1], images[2:]


def get_dataloader(
    dataset_dir: str = "/homedtcl/nrakotonirina/datasets/isc2021",
    num_distractors: int = 63,
    batch_size: int = 32,
    image_size: int = 32,
    num_workers: int = 8,
):

    ds = ISC2021Dataset(
        dataset_dir=dataset_dir,
        num_distractors=num_distractors,
        transform=get_transform(image_size),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    dataset_dir = "/homedtcl/nrakotonirina/datasets/isc2021"
    loader = get_dataloader(
        dataset_dir=dataset_dir, num_distractors=6, bacth_size=8, num_workers=0
    )

    for i, sample in enumerate(loader):
        print(sample, len(sample[-1]))
        break
