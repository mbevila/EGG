# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time

import torch
from transformers import AdamW

import egg.core as core
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.baseline.game import build_game
from egg.zoo.emergent_captioner.dataloaders.coco_dataloader import CocoWrapper
from egg.zoo.emergent_captioner.dataloaders.flickr_dataloader import FlickrWrapper
from egg.zoo.emergent_captioner.utils import (
    get_sha,
    log_stats,
    store_job_and_task_id,
)


def main(params):
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument(
        "--recv_clip_model",
        choices=["ViT-B/16", "ViT-B/32"],
        default="ViT-B/32",
    )
    parser.add_argument(
        "--dataset",
        choices=["coco", "flickr"],
        default="coco",
    )
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=8)

    opts = core.init(arg_parser=parser, params=params)

    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    data_kwargs = dict(
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        num_workers=opts.num_workers,
    )

    if opts.dataset == "coco":
        coco_wrapper = CocoWrapper()
        test_loader = coco_wrapper.get_split(split="test", **data_kwargs)
    elif opts.dataset == "flickr":
        flickr_wrapper = FlickrWrapper()
        test_loader = flickr_wrapper.get_split(split="test", **data_kwargs)
    elif opts.dataset == "conceptual_captions":
        raise NotImplementedError

    game = build_game(opts)

    trainer = core.Trainer(
        game=game,
        optimizer=AdamW(game.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
    )

    trainer.game.test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, True, False
    )
    _, interaction = trainer.eval(test_loader)

    log_stats(interaction, "TEST SET")

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
