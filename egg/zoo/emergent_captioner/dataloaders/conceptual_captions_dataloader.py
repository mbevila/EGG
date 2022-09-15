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
import torch.distributed as dist
from torchvision.datasets import VisionDataset
from torch.utils.data.distributed import DistributedSampler

from egg.zoo.emergent_captioner.dataloaders.utils import get_transform


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
            with open(img_path, "rb") as f:
                image = Image.open(f).convert("RGB")
        except UnidentifiedImageError:
            return self.__getitem__(random.choice(self.samples))

        if self.transform:
            image = self.transform(image)

        aux = {
            "img_id": fname,
            "caption": caption,
            "all_captions": caption,
        }
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
            sampler = DistributedSampler(
                ds, shuffle=split != "test", drop_last=True, seed=seed
            )

        print(f"shuff is {split != 'test' and sampler is None}")
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split != "test" and sampler is None,
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
        image_size=224,
        num_workers=8,
    )

    for i, (_, idx, _, aux) in enumerate(dl):
        if i == 5000:
            break
        if i % 1000 == 0:
            print("eleem", i)
