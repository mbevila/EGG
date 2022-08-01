# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.receiver import ClipReceiver
from egg.zoo.emergent_captioner.finetuning.sender import ClipCapSender


def accuracy_loss(
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
    return acc, {"acc": acc}


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

    loss = F.cross_entropy(receiver_output, labels, reduction="none")
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return loss, {"acc": acc}


def similarity_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    raise NotImplementedError


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class ReinforceCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(ReinforceCaptionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.train_logging_strategy = (
            LoggingStrategy().minimal()
            if train_logging_strategy is None
            else train_logging_strategy
        )
        self.test_logging_strategy = (
            LoggingStrategy().minimal()
            if test_logging_strategy is None
            else test_logging_strategy
        )

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        bs_message, greedy_message, log_prob = self.sender(sender_input, aux_input)

        receiver_output_bms = self.receiver(bs_message, receiver_input, aux_input)
        bs_reward, aux_info = self.loss(
            sender_input,
            bs_message,
            receiver_input,
            receiver_output_bms,
            labels,
            aux_input,
        )
        with torch.no_grad():
            receiver_output_greedy = self.receiver(
                greedy_message, receiver_input, aux_input
            )
            greedy_reward, aux_info = self.loss(
                sender_input,
                greedy_message,
                receiver_input,
                receiver_output_greedy,
                labels,
                aux_input,
            )

        # ignoring entropy for now
        reward = bs_reward - greedy_reward.detach()
        policy_loss = reward * log_prob

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=bs_message,
            receiver_output=receiver_output_bms.detach(),
            message_length=None,
            aux=aux_info,
        )

        return policy_loss.mean(), interaction


def build_game(opts):
    sender = ClipCapSender(
        clip_prefix_tokens=opts.clip_prefix_tokens,
        clip_model=opts.sender_clip_model,
        clip_cap_path=opts.clipcap_model_path,
        beam_size=opts.beam_size,
    )
    sender.setup_clipcap(opts.clip_prefix_tokens, opts.batch_size)

    receiver = ClipReceiver(clip_model=opts.recv_clip_model)

    # TODO add option to use other losses
    # remember that with non-diff loss you should use a wrapper around recv
    game = ReinforceCaptionGame(sender, receiver, discriminative_loss)
    return game
