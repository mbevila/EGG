# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import random
import zlib
from pathlib import Path
from typing import Callable, Optional
from PIL import Image, UnidentifiedImageError

import torch
from torchvision.datasets import VisionDataset

from egg.zoo.emergent_captioner.dataloaders.utils import get_transform


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ConceptualCaptionsDataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = "/private/home/rdessi/ConceptualCaptions",
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super(ConceptualCaptionsDataset, self).__init__(
            dataset_dir, transform=transform
        )
        self.dataset_dir = Path(dataset_dir)
        assert split in ["train", "test"]

        if split == "train":
            annotations_file = "Train_GCC-training.tsv"
            self.image_folder = self.dataset_dir / "training"
        else:
            annotations_file = "Validation_GCC-1.1.0-Validation.tsv"
            self.image_folder = self.dataset_dir / "validation"

        all_images = {f for f in os.listdir(self.image_folder) if f[0].isdigit()}

        self.samples = []
        with open(self.dataset_dir / annotations_file) as fd:
            reader = csv.reader(fd, delimiter="\t")
            for i, (caption, url) in enumerate(reader):
                filename = f"{i}_{zlib.crc32(url.encode('utf-8')) & 0xFFFFFFFF}"
                if filename in all_images:
                    self.samples.append((caption, filename))
        self.idx = -1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        caption, fname = self.samples[index]

        img_path = self.image_folder / fname

        try:
            image = pil_loader(img_path)
        except UnidentifiedImageError:
            return self.__getitem__(random.choice(self.samples))

        if self.transform:
            image = self.transform(image)

        aux = {
            "image_ids": fname,
            "caption": caption,
            "all_captions": caption,
        }
        return image, torch.tensor([index]), image, aux


def get_dataloader(
    dataset_dir: str = "/private/home/rdessi/ConceptualCaptions",
    batch_size: int = 32,
    image_size: int = 32,
    split: str = "train",
    num_workers: int = 8,
):

    ds = ConceptualCaptionsDataset(
        dataset_dir=dataset_dir, split=split, transform=get_transform(image_size)
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=split != "test",
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    dataset_dir = "/private/home/rdessi/ConceptualCaptions"
    dl = get_dataloader(
        dataset_dir=dataset_dir,
        split="test",
        batch_size=1,
        num_workers=0,
    )

    for i, (_, idx, _, aux) in enumerate(dl):
        if i == 5000:
            break
        if i % 1000 == 0:
            print("eleem", i)
