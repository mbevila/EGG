# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import egg.core as core
from egg.core import Callback, ConsoleLogger, Interaction
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.dataloaders.flickr_dataloader import get_dataloader
from egg.zoo.emergent_captioner.finetuning.game import build_game
from egg.zoo.emergent_captioner.finetuning.opts import get_common_opts
from egg.zoo.emergent_captioner.utils import (
    dump_interaction,
    get_sha,
    log_stats,
    # print_grad_info,
    store_job_and_task_id,
)


class ModelSaver(Callback):
    def save_clipclap_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                model_name = f"clip_clap_model_{epoch if epoch else 'final'}.pt"

                torch.save(
                    self.trainer.game.sender.clipcap.state_dict(),
                    self.trainer.checkpoint_path / model_name,
                )

    def on_epoch_end(self, loss: float, _logs: Interaction, epoch: int):
        self.save_clipclap_model(epoch=epoch)

    def on_train_end(self):
        self.save_clipclap_model()


def main(params):
    start = time.time()
    opts = get_common_opts(params=params)

    store_job_and_task_id(opts)
    print(opts)
    print(get_sha())

    if not opts.distributed_context.is_distributed and opts.debug:
        breakpoint()

    train_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="train",
        num_workers=opts.num_workers,
    )
    """
    val_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="val",
        num_workers=opts.num_workers,
    )
    """

    game = build_game(opts)
    # print_grad_info(game.sender)

    name2opt = {"adam": torch.optim.Adam, "adamw": AdamW}

    optimizer = name2opt[opts.opt.lower()](game.sender.parameters(), lr=opts.lr)

    scheduler = None
    if opts.opt_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=opts.warmup_steps,
            num_training_steps=opts.n_epochs * len(train_loader),
        )

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        optimizer_scheduler=scheduler,
        train_data=train_loader,
        # validation_data=val_loader,
        callbacks=[ConsoleLogger(as_json=True, print_train_loss=True), ModelSaver()],
        debug=opts.debug,
    )

    trainer.train(opts.n_epochs)

    test_loader = get_dataloader(
        dataset_dir=opts.dataset_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="test",
        num_workers=opts.num_workers,
    )

    trainer.game.test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, True, False
    )
    _, test_interaction = trainer.eval(test_loader)

    log_stats(test_interaction, "TEST SET")
    dump_interaction(test_interaction, opts)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    # torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
