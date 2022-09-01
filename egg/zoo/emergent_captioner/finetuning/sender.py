# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from types import MethodType
from typing import Any, Dict, Tuple

import clip
import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from undecorated import undecorated

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ClipCapModel(nn.Module):
    def __init__(
        self,
        clip_prefix_size: int,
        clip_prefix_tokens: int,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapModel, self).__init__()

        assert beam_size > 1

        self.clip_prefix_tokens = clip_prefix_tokens

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id

        generate_with_grad = undecorated(self.gpt.generate)
        self.gpt.generate_with_grad = MethodType(generate_with_grad, self.gpt)

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.gpt.config.pad_token_id

        self.beam_size = beam_size
        self.max_len = max_len

        if clip_prefix_tokens > 10:  # not enough memory
            input_dim = clip_prefix_size
            output_dim = self.gpt_embedding_size * clip_prefix_tokens
            self.clip_project = nn.Linear(input_dim, output_dim)
        else:
            input_dim = clip_prefix_size
            hidden_dim = (self.gpt_embedding_size * clip_prefix_tokens) // 2
            output_dim = self.gpt_embedding_size * clip_prefix_tokens
            self.clip_project = MLP((input_dim, hidden_dim, output_dim))

    def setup_input_output_embeddings(self, bsz, prefix_len):
        self.gpt.resize_token_embeddings(len(self.tokenizer) + (prefix_len * bsz))

    def get_greedy_baseline(self):
        generated_ids = self.gpt.generate(
            self.cached_input_ids, max_length=self.max_len + self.clip_prefix_tokens
        )

        decoded = self.tokenizer.batch_decode(
            generated_ids[:, self.clip_prefix_tokens :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        captions = []
        for caption in decoded:
            text = caption + "."
            filtered = text[: text.index(".") + 1]
            captions.append(filtered)

        return captions

    def forward(self, image_feats, aux_input=None):
        bsz = image_feats.shape[0]

        prefix_embed = self.clip_project(image_feats)

        prefix_embed = prefix_embed.view(bsz, self.clip_prefix_tokens, -1)
        bsz, prefix_len, h_dim = prefix_embed.shape

        bias = torch.Tensor(
            [0.0] * len(self.tokenizer) + [float("-inf")] * bsz * prefix_len
        ).to(prefix_embed.device)
        self.gpt.get_output_embeddings().bias = nn.Parameter(bias, requires_grad=False)

        prefix_embed_flat = prefix_embed.view(-1, h_dim)
        start, end = len(self.tokenizer), len(self.tokenizer) + (bsz * prefix_len)
        self.gpt.get_input_embeddings().weight.data[start:end] = prefix_embed_flat
        input_ids = torch.arange(start, end)
        input_ids = input_ids.view(bsz, prefix_len).to(prefix_embed.device)
        self.cached_input_ids = input_ids

        generated = self.gpt.generate_with_grad(
            input_ids,
            max_length=self.max_len + self.clip_prefix_tokens,
            output_scores=True,
            return_dict_in_generate=True,
        )

        max_k = generated.sequences.size(1) - prefix_len
        zero_mask = generated.sequences[:, prefix_len:] == self.tokenizer.encode(".")[0]
        zero_mask = zero_mask.cumsum(dim=1) > 0
        msg_lengths = max_k - (zero_mask).sum(dim=1)
        msg_lengths.add_(1).clamp_(max=max_k)

        log_probs = generated.scores
        log_probs = torch.stack(log_probs, dim=1)[..., : len(self.tokenizer)]

        entropy = torch.stack(
            [
                Categorical(probs=log_probs[:, timestep].detach().softmax(-1)).entropy()
                for timestep in range(log_probs.shape[1])
            ],
            dim=1,
        )

        entropy *= (~zero_mask).long()
        entropy = entropy.sum(-1) / msg_lengths

        log_probs = torch.topk(log_probs, largest=True, k=1, dim=-1)[0].squeeze()
        log_probs *= (~zero_mask).long()
        log_probs = log_probs.sum(-1) / msg_lengths

        decoded = self.tokenizer.batch_decode(
            generated.sequences[:, self.clip_prefix_tokens :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        captions = []
        for caption in decoded:
            text = caption + "."
            filtered = text[: text.index(".") + 1]
            captions.append(filtered)

        return captions, log_probs, entropy, msg_lengths

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.gpt.named_parameters()

    def parameters(self, recurse: bool = True):
        return self.gpt.parameters()


class ClipCapSender(nn.Module):
    def __init__(
        self,
        clip_prefix_tokens: int,
        clip_model: str,
        clipcap_path: str,
        beam_size: int = 5,
        max_len: int = 20,
    ):
        super(ClipCapSender, self).__init__()

        assert max_len < 75  # Clip maximum context size

        self.clip_vit = clip.load(clip_model)[0].visual
        convert_models_to_fp32(self.clip_vit)
        self.clip_vit.eval()

        for p in self.clip_vit.parameters():
            p.requires_grad = False

        self.clipcap = ClipCapModel(
            clip_prefix_size=self.clip_vit.output_dim,
            clip_prefix_tokens=clip_prefix_tokens,
            beam_size=beam_size,
            max_len=max_len,
        )
        if clipcap_path is not None:
            self.clipcap.load_state_dict(torch.load(clipcap_path))

    def setup_clipcap(self, batch_size, n_prefix_tokens):
        self.clipcap.setup_input_output_embeddings(batch_size, n_prefix_tokens)

    def encode_images(self, images: torch.Tensor):
        return self.clip_vit(images)

    def get_greedy_baseline(self):
        return self.clipcap.get_greedy_baseline()

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        image_feats = self.encode_images(images)
        captions, log_probs, entropy, msg_lengths = self.clipcap(image_feats, aux_input)
        return captions, log_probs, entropy, msg_lengths

    def named_parameters(self, prefix="", recurse: bool = True):
        return self.clipcap.named_parameters()

    def parameters(self, recurse: bool = True):
        return self.clipcap.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clip_vit.eval()
        self.clipcap.train(mode)
        return self
