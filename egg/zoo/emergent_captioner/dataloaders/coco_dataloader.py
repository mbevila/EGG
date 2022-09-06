# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.datasets import CocoCaptions

from egg.zoo.emergent_captioner.dataloaders.utils import get_transform


class CocoDataset(CocoCaptions):
    def __getitem__(self, idx):
        image, captions = super().__getitem__(idx)

        aux = {"all_captions": captions, "caption": captions[0]}

        return image, torch.tensor([idx]), image, aux


def get_dataloader(
    image_dir: str,
    metadata_dir: str,
    split: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 8,
):
    ds = CocoDataset(
        root=image_dir, annFile=metadata_dir, transform=get_transform(image_size)
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split != "test",
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    image_dir = "/datasets01/COCO/060817/val2014"
    metadata_dir = "/datasets01/COCO/060817/annotations/captions_val2014.json"
    dl = get_dataloader(
        image_dir=image_dir,
        metadata_dir=metadata_dir,
        split="val",
        batch_size=8,
        num_workers=0,
    )
