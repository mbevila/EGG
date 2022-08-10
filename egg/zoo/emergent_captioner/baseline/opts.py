# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import egg.core as core


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument("--dataset_dir", default="/checkpoint/rdessi/datasets/flickr30k")
    group.add_argument("--image_size", type=int, default=224, help="Image size")
    group.add_argument("--num_workers", type=int, default=8)


def get_game_opts(parser):
    group = parser.add_argument_group("game options")

    group.add_argument(
        "--sender",
        choices=["human", "clipcap"],
        default="human",
    )
    group.add_argument(
        "--clipcap_model_path",
        default="/private/home/rdessi/EGG/egg/zoo/emergent_captioner/clipclap_models/conceptual_weights.pt",
    )
    group.add_argument(
        "--sender_clip_model",
        choices=["ViT-B/16", "ViT-B/32", "RN50x4"],
        default="ViT-B/32",
    )
    group.add_argument(
        "--recv_clip_model",
        choices=["ViT-B/16", "ViT-B/32", "RN50x4"],
        default="ViT-B/32",
    )
    group.add_argument(
        "--clip_prefix_tokens",
        type=int,
        default=10,
        help="Number of prefix tokens generated from a clip image embedding",
    )
    group.add_argument(
        "--use_beam_search",
        action="store_true",
        default=False,
        help="Use beam search decoding when generating captions",
    )
    group.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Number of beams when using beam search decoding",
    )


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )

    get_data_opts(parser)
    get_game_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
