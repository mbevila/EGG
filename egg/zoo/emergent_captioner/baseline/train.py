# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from transformers import AdamW

import egg.core as core
from egg.zoo.emergent_captioner.baseline.game import build_game
from egg.zoo.emergent_captioner.dataloaders.flickr_dataloader import get_dataloader
from egg.zoo.emergent_captioner.baseline.opts import get_common_opts
from egg.zoo.emergent_captioner.utils import (
    get_sha,
    log_stats,
    store_job_and_task_id,
)


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    test_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="test",
        num_workers=opts.num_workers,
    )

    game = build_game(opts)

    trainer = core.Trainer(
        game=game,
        optimizer=AdamW(game.parameters(), lr=opts.lr),
        train_data=None,
        debug=opts.debug,
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
