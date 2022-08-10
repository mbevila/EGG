# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license fod in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional
from PIL import Image

import torch
from torchvision.datasets import VisionDataset

from egg.zoo.emergent_captioner.dataloaders.utils import get_transform


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse

    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this
                                    phrase belongs to

    """
    with open(fn, "r") as f:
        sentences = f.read().split("\n")

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == "]":
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(" ".join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == "[":
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split("/")
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {"sentence": " ".join(words), "phrases": []}
        for index, phrase, p_id, p_type in zip(
            first_word, phrases, phrase_id, phrase_type
        ):
            sentence_data["phrases"].append(
                {
                    "first_word_index": index,
                    "phrase": phrase,
                    "phrase_id": p_id,
                    "phrase_type": p_type,
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall("size")[0]
    anno_info = {"boxes": {}, "scene": [], "nobox": []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall("object"):
        for names in object_container.findall("name"):
            box_id = names.text
            box_container = object_container.findall("bndbox")
            if len(box_container) > 0:
                if box_id not in anno_info["boxes"]:
                    anno_info["boxes"][box_id] = []
                xmin = int(box_container[0].findall("xmin")[0].text) - 1
                ymin = int(box_container[0].findall("ymin")[0].text) - 1
                xmax = int(box_container[0].findall("xmax")[0].text) - 1
                ymax = int(box_container[0].findall("ymax")[0].text) - 1
                anno_info["boxes"][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall("nobndbox")[0].text)
                if nobndbox > 0:
                    anno_info["nobox"].append(box_id)

                scene = int(object_container.findall("scene")[0].text)
                if scene > 0:
                    anno_info["scene"].append(box_id)

    return anno_info


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Flickr30kDataset(VisionDataset):
    def __init__(
        self,
        dataset_dir: str = "/checkpoint/rdessi/datasets/flickr30k",
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        super(Flickr30kDataset, self).__init__(dataset_dir, transform=transform)
        dataset_dir = Path(dataset_dir)
        with open(dataset_dir / f"{split}.txt") as f:
            split_images = set(f.read().splitlines())

        ann_paths = glob.iglob(f"{os.path.expanduser(dataset_dir)}/Annotations/*xml")
        self.samples = []
        for ann_path in ann_paths:
            image_id = Path(ann_path).stem
            if image_id not in split_images:
                continue

            img_path = Path(dataset_dir) / "Images" / f"{image_id}.jpg"
            anns = get_annotations(ann_path)
            sents = get_sentence_data(dataset_dir / "Sentences" / f"{image_id}.txt")

            self.samples.append((img_path, anns, sents))

            if len(self.samples) >= len(split_images):
                break

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, anns, sents = self.samples[index]

        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)

        # caption = random.choice(sents)["sentence"]
        all_captions = [elem["sentence"] for elem in sents]

        aux = {
            "image_ids": torch.Tensor([int(img_path.stem)]),
            "caption": all_captions[0],
            "all_captions": all_captions,
        }
        return image, torch.zeros(1), image, aux


def get_dataloader(
    dataset_dir: str = "/checkpoint/rdessi/datasets/flickr30k",
    batch_size: int = 32,
    image_size: int = 32,
    split: str = "train",
    num_workers: int = 8,
):

    ds = Flickr30kDataset(
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
    dataset_dir = "/checkpoint/rdessi/datasets/flickr30k"
    dl = get_dataloader(
        dataset_dir=dataset_dir,
        split="test",
        batch_size=8,
        num_workers=0,
    )
