# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import GPT2Tokenizer

from egg.core.baselines import MeanBaseline, NoBaseline
from egg.core.interaction import LoggingStrategy
from egg.zoo.emergent_captioner.receiver import ClipReceiver
from egg.zoo.emergent_captioner.finetuning.losses import (
    AccuracyLoss,
    DiscriminativeLoss,
    SimilarityLoss,
)
from egg.zoo.emergent_captioner.finetuning.sender import ClipCapSender, BLIPSender

dataset2paths = {
    "flickr": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/test_flickr.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/test_flickr.nns.pt",
    ),
    "coco": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/train_coco.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/test_coco.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/train_coco.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/coco/test_coco.nns.pt",
    ),
    "conceptual": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/test_conceptual.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/train_conceptual.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/conceptual/test_conceptual.nns.pt",
    ),
    "nocaps_in-domain": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/indomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/indomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/indomain.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/indomain.nns.pt",
    ),
    "nocaps_near-domain": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/neardomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/neardomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/neardomain.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/neardomain.nns.pt",
    ),
    "nocaps_out-domain": (
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/outdomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/outdomain.emb.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/outdomain.nns.pt",
        "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/nocaps/outdomain.nns.pt",
    ),
}


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
    return -acc, {"acc": acc}


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
    _labels,
    _aux_input,
):
    batch_size = receiver_output.shape[0]
    labels = torch.arange(batch_size, device=receiver_output.device)

    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    return receiver_output, {"acc": acc}


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()


class ReinforceCaptionGame(nn.Module):
    def __init__(
        self,
        sender: nn.Module,
        receiver: nn.Module,
        loss: Callable,
        sender_entropy_coeff: float = 0.0,
        baseline: str = "no",
        train_logging_strategy: LoggingStrategy = None,
        test_logging_strategy: LoggingStrategy = None,
    ):
        super(ReinforceCaptionGame, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

        self.baseline_name = baseline
        self.baseline = {"no": NoBaseline, "mean": MeanBaseline}[baseline]()

        self.sender_entropy_coeff = sender_entropy_coeff

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

        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        captions, log_prob, entropy, msg_lengths = self.sender(sender_input, aux_input)

        with torch.no_grad():
            receiver_output = self.receiver(captions, receiver_input, aux_input)
            loss, aux_info = self.loss(
                sender_input,
                captions,
                receiver_input,
                receiver_output,
                labels,
                aux_input,
            )

        weighted_entropy = entropy * self.sender_entropy_coeff

        baseline = self.baseline.predict(loss.detach())

        policy_loss = ((loss.detach() - baseline) * log_prob).mean()

        optimized_loss = policy_loss - weighted_entropy

        if self.training:
            self.baseline.update(loss)

        aux_info["sender_entropy"] = entropy

        """
        captions = self.tokenizer(
            captions, padding="max_length", max_length=300
        ).input_ids
        captions = torch.tensor(captions)

        gt_captions = self.tokenizer(
            aux_input["caption"], padding="max_length", max_length=300
        ).input_ids
        aux_input["caption"] = torch.tensor(gt_captions)
        """

        logging_strategy = (
            self.train_logging_strategy if self.training else self.test_logging_strategy
        )
        interaction = logging_strategy.filtered_interaction(
            sender_input=sender_input,
            labels=labels,
            receiver_input=receiver_input,
            aux_input=aux_input,
            message=captions,
            receiver_output=receiver_output.detach(),
            message_length=msg_lengths,
            aux=aux_info,
        )

        return optimized_loss.mean(), interaction


def get_loss(
    loss_type: str,
    num_hard_negatives: int,
    in_batch_negatives: bool,
    dataset: str,
    split: str,
    num_return_sequences: int = 1,
    test_w_negatives: bool = False,
    logit_scale: float = 1.0,
):
    if loss_type.lower() != "discriminative":
        assert RuntimeError("loss {loss_type} not implemented yet")

    assert dataset in ["flickr", "coco", "conceptual", "nocaps"]
    if dataset == "nocaps":
        dataset == "_".join([dataset, split])

    train_emb, test_emb, train_nns, test_nns = dataset2paths[dataset.lower()]

    name2loss = {
        "discriminative": DiscriminativeLoss,
        "accuracy": AccuracyLoss,
        "similarity": SimilarityLoss,
    }

    loss_cls = name2loss.get(loss_type.lower(), None)
    assert loss_cls, f"cannot recognize loss {loss_type}"

    loss = loss_cls(
        train_emb,
        train_nns,
        test_emb,
        test_nns,
        num_hard_negatives,
        in_batch_negatives,
        test_w_negatives,
        num_return_sequences,
        logit_scale,
    )
    return loss


def build_game(opts):

    if opts.captioner_model == 'clipcap':
        sender = ClipCapSender(
            clip_model=opts.sender_clip_model,
            clipcap_path=opts.clipcap_model_path,
            freeze_clipcap_mapper=opts.freeze_clipcap_mapper,
            num_return_sequences=opts.num_return_sequences,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    elif opts.captioner_model == 'blip':
        sender = BLIPSender(
            blip_model=opts.blip_model,
            freeze_image_encoder=opts.blip_freeze_img_encoder,
            num_return_sequences=opts.num_return_sequences,
            do_sample=opts.do_sample,
            beam_size=opts.beam_size,
            max_len=opts.max_len,
        )
    else:
        raise NotImplementedError

    if opts.set_buffer_size > 0:
        sender.unpatch_model()
        sender.patch_model(opts.set_buffer_size)

    receiver = ClipReceiver(clip_model=opts.recv_clip_model)

    test_logging_strategy = LoggingStrategy(
        False, False, True, True, True, False, False
    )

    # remember that with non-diff losses you should use a wrapper around recv
    loss = get_loss(
        loss_type=opts.loss_type,
        num_hard_negatives=opts.num_hard_negatives,
        in_batch_negatives=opts.in_batch_negatives,
        dataset=opts.dataset,
        split=opts.split,
        num_return_sequences=opts.num_return_sequences,
        test_w_negatives=opts.test_w_negatives,
        logit_scale=receiver.clip.logit_scale,
    )
    game = ReinforceCaptionGame(
        sender=sender,
        receiver=receiver,
        loss=loss,
        baseline=opts.baseline,
        sender_entropy_coeff=opts.sender_entropy_coeff,
        test_logging_strategy=test_logging_strategy,
    )
    return game
