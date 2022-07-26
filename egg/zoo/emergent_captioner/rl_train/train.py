# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from transformers import AdamW, get_linear_schedule_with_warmup

import egg.core as core
from egg.core import Callback, ConsoleLogger, Interaction
from egg.zoo.contextual_game.game import build_game
from egg.zoo.contextual_game.dataloaders.flickr_dataloader import get_dataloader
from egg.zoo.contextual_game.opts import get_common_opts
from egg.zoo.contextual_game.utils import get_sha, store_job_and_task_id


def print_grad_info(model):
    grad, no_grad = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            no_grad.append(name)
            continue
        grad.append(name)
    print(f"GRAD {grad}")
    print(f"NO GRAD {no_grad}")


class ModelSaver(Callback):
    def save_clipclap_model(self, epoch=""):
        if hasattr(self.trainer, "checkpoint_path"):
            if (
                self.trainer.checkpoint_path
                and self.trainer.distributed_context.is_leader
            ):
                self.trainer.checkpoint_path.mkdir(exist_ok=True, parents=True)
                model_name = f"clip_clap_model_{epoch if epoch else 'final'}.pt"

                # TODO check accessing to clipclap model is done correctly
                torch.save(
                    self.trainer.game.sender.clipclap_model.state_dict(),
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
        image_dir=opts.image_dir,
        metadata_dir=opts.metadata_dir,
        batch_size=opts.batch_size,
        image_size=opts.image_size,
        split="train",
        num_workers=opts.num_workers,
        is_distributed=opts.distributed_context.is_distributed,
        seed=opts.random_seed,
    )

    game = build_game(opts)
    print_grad_info(game)

    optimizer = AdamW(game.parameters(), lr=opts.lr)
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
        callbacks=[ConsoleLogger(as_json=True, print_train_loss=True), ModelSaver()],
        debug=opts.debug,
    )

    trainer.train(opts.n_epochs)

    end = time.time()
    print(f"| Run took {end - start:.2f} seconds")
    print("| FINISHED JOB")


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_deterministic(True)
    import sys

    main(sys.argv[1:])
