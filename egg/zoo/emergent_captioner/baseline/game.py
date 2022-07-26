# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer

from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.baseline.sender import ClipCapSender, HumanCaptionSender
from egg.zoo.emergent_captioner.receiver import ClipReceiver


class ZeroShotCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        logging_strategy: LoggingStrategy = None,
    ):
        super(ZeroShotCaptionGame, self).__init__()

        self.train_logging_strategy = LoggingStrategy.minimal()
        self.test_logging_strategy = (
            LoggingStrategy.minimal() if logging_strategy is None else logging_strategy
        )

        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        message = self.sender(sender_input, aux_input)
        message_length = torch.Tensor([len(x) for x in message]).int()

        receiver_output = self.receiver(message, receiver_input, aux_input)

        loss, aux_info = self.loss(
            sender_input, message, receiver_input, receiver_output, labels, aux_input
        )

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )

        # Note that this is not what is processed by the clip receiver since clip
        #  automatically truncate text after 75 tokens
        tokenized_text = self.tokenizer(message, padding=True, return_tensors="pt")

        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=tokenized_text,
            receiver_output=receiver_output.detach(),
            message_length=message_length,
            aux=aux_info,
        )
        return loss.mean(), interaction


def discriminative_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    _labels,
    _aux_input,
):
    batch_size = receiver_output.shape[0]
    labels = torch.arange(batch_size, device=receiver_output.device)

    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return torch.zeros(1), {"acc": acc}


def build_game(opts):

    if opts.sender == "human":
        sender = HumanCaptionSender()
    elif opts.sender == "clipcap":
        sender = ClipCapSender(
            clip_prefix_tokens=opts.clip_prefix_tokens,
            clip_model=opts.sender_clip_model,
            clip_cap_path=opts.clipcap_model_path,
            use_beam_search=opts.use_beam_search,
            beam_size=opts.beam_size,
        )

        sender.setup_clipcap(opts.clip_prefix_tokens, opts.batch_size)
    receiver = ClipReceiver(clip_model=opts.recv_clip_model)

    loss = discriminative_loss

    game = ZeroShotCaptionGame(sender, receiver, loss)
    return game
