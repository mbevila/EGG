# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

import torch

import egg.core as core


def store_job_and_task_id(opts):
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", None)
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None or task_id is None:
        job_id = os.environ.get("SLURM_JOB_ID", uuid.uuid4())
        task_id = os.environ.get("SLURM_PROCID", 0)

    opts.job_id = job_id
    opts.task_id = task_id
    return job_id, task_id


def log_stats(interaction, mode):
    dump = dict((k, v.mean().item()) for k, v in interaction.aux.items())
    dump.update(dict(mode=mode))
    print(json.dumps(dump), flush=True)


def dump_interaction(interaction, opts):
    if opts.checkpoint_dir:
        output_path = Path(opts.checkpoint_dir) / "interactions"
        output_path.mkdir(exist_ok=True, parents=True)
        interaction_name = f"interaction_{opts.job_id}_{opts.task_id}"

        interaction.aux_input["args"] = opts
        torch.save(interaction, output_path / interaction_name)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_data_opts(parser):
    group = parser.add_argument_group("data options")
    group.add_argument("--image_size", type=int, default=224, help="Image size")
    group.add_argument(
        "--image_dir", default="/private/home/rdessi/imagecode/data/images"
    )
    group.add_argument(
        "--metadata_dir",
        default="/private/home/rdessi/exp_EGG/egg/zoo/contextual_game/dataset",
    )
    group.add_argument("--num_workers", type=int, default=8)


def get_clipclap_opts(parser):
    group = parser.add_argument_group("clipclap opts")
    group.add_argument("--clipclap_model_path", default=None)
    group.add_argument("--mapping_type", choices=["mlp", "transformer"], default="mlp")
    group.add_argument("--clip_prefix_tokens", type=int, default=10)
    group.add_argument("--constant_prefix_tokens", type=int, default=10)
    group.add_argument("--num_transformer_layers", type=int, default=8)
    group.add_argument(
        "--clip_model", choices=["ViT-B/32", "RN50x4"], default="ViT-B/32"
    )
    group.add_argument("--use_beam_search", action="store_true", default=False)
    group.add_argument("--num_beams", type=int, default=5)
    group.add_argument("--prefix_only", action="store_true", default=False)


def get_common_opts(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run the game with pdb enabled",
    )
    parser.add_argument("--warmup_steps", type=int, default=5000)

    get_data_opts(parser)
    get_clipclap_opts(parser)

    opts = core.init(arg_parser=parser, params=params)
    return opts
