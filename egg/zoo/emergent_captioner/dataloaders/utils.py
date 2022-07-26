# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

from PIL import Image

from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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
