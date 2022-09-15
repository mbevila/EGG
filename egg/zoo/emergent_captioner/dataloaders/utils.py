# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import math
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class MyDistributedSampler(DistributedSampler):
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        begin = self.rank * self.num_samples
        end = begin + self.num_samples
        indices = indices[begin:end]
        assert len(indices) == self.num_samples

        return iter(indices)


def get_transform(image_size: int):
    def _convert_image_to_rgb(image: Image.Image):
        return image.convert("RGB")

    transformations = [
        transforms.Resize(image_size, interpolation=BICUBIC),
        transforms.CenterCrop(image_size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
    return transforms.Compose(transformations)
