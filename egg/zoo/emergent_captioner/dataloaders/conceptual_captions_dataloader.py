# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import csv
import os

# import random
import zlib
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
import torch.distributed as dist
from timebudget import timebudget
from torchvision.datasets import VisionDataset

from egg.zoo.emergent_captioner.dataloaders.utils import (
    get_transform,
    MyDistributedSampler,
)


class ConceptualCaptionsDataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = None,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        if dataset_dir is None:
            dataset_dir = "/private/home/rdessi/ConceptualCaptions"
        super(ConceptualCaptionsDataset, self).__init__(
            dataset_dir, transform=transform
        )
        self.dataset_dir = Path(dataset_dir)
        assert split in ["train", "test"]

        if split == "train":
            annotations_file = "train_conceptual_captions_paths.txt"
            self.image_folder = self.dataset_dir / "training"
        else:
            # annotations_file = "train_conceptual_captions_paths.txt"
            annotations_file = "Validation_GCC-1.1.0-Validation.tsv"
            self.image_folder = self.dataset_dir / "validation"

        with timebudget("0"):
            all_images = {
                f.name for f in os.scandir(self.image_folder) if f.name[0].isdigit()
            }

        with timebudget("1"):
            self.samples = []
            with open(self.dataset_dir / annotations_file) as fd:
                reader = csv.reader(fd, delimiter="\t")
                for i, (caption, url) in enumerate(reader):
                    filename = f"{i}_{zlib.crc32(url.encode('utf-8')) & 0xFFFFFFFF}"
                    if filename in all_images:
                        self.samples.append((caption, filename))
        with open("/private/home/rdessi/test_conceptual_captions_paths.txt", "w") as f:
            for sample in self.samples:
                f.write(f"{sample[1].strip()}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        caption, fname = self.samples[index]

        img_path = self.image_folder / fname

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        if self.transform:
            image = self.transform(image)

        aux = {"img_id": torch.tensor([fname])}
        return image, torch.tensor([index]), image, aux


class ConceptualCaptionsWrapper:
    def __init__(self, dataset_dir: str):
        if dataset_dir is None:
            dataset_dir = "/private/home/rdessi/ConceptualCaptions"
        self.dataset_dir = Path(dataset_dir)

    def get_split(
        self,
        split: str,
        batch_size: int,
        image_size: int,
        num_workers: int = 8,
        seed: int = 111,
    ):
        ds = ConceptualCaptionsDataset(
            dataset_dir=self.dataset_dir,
            split=split,
            transform=get_transform(image_size),
        )
        sampler = None
        if dist.is_initialized():
            sampler = MyDistributedSampler(
                ds, shuffle=split != "test", drop_last=True, seed=seed
            )

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # split != "test" and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return loader


if __name__ == "__main__":
    dataset_dir = "/private/home/rdessi/ConceptualCaptions"
    wrapper = ConceptualCaptionsWrapper(dataset_dir)
    dl = wrapper.get_split(
        split="test",
        batch_size=1,
        image_size=32,
        num_workers=8,
    )
    print("len data", len(dl.dataset))
    with timebudget("all"):
        for i, elem in enumerate(dl):
            print("eleem", i)
