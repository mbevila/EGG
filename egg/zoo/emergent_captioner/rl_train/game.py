# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import clip
import torch

from egg.zoo.emergent_captioner.archs import ClipCaptionModel, ClipReceiver, RLGame


def discriminative_loss(
    _sender_input,
    _message,
    _receiver_input,
    receiver_output,
    labels,
    _aux_input,
):
    loss = torch.zeros(1).to(receiver_output.device)
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


def build_game(opts):
    clip_model = clip.load(opts.clip_model)[0]
    convert_models_to_fp32(clip_model)

    sender = ClipCaptionModel(
        model_path=opts.clipclap_model_path,
        mapping_type=opts.mapping_type,
        constant_prefix_tokens=opts.constant_prefix_tokens,
        clip_prefix_tokens=opts.clip_prefix_tokens,
        clip_prefix_size=clip_model.visual.output_dim,
        num_layers=opts.num_transformer_layers,
        clip_model=opts.clip_model,
        use_beam_search=opts.use_beam_search,
        num_beams=opts.num_beams,
    )

    receiver = ClipReceiver(clip_model)

    game = RLGame(sender, receiver, discriminative_loss)

    return game
