# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import clip
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
    ):
        super(ClipCapModel, self).__init__()

        self.clip_prefix_tokens = clip_prefix_tokens

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.gpt.config.pad_token_id

        self.beam_size = beam_size

        if clip_prefix_tokens > 10:  # not enough memory
            input_dim = clip_prefix_size
            output_dim = self.gpt_embedding_size * clip_prefix_tokens
            self.clip_project = nn.Linear(input_dim, output_dim)
        else:
            input_dim = clip_prefix_size
            hidden_dim = (self.gpt_embedding_size * clip_prefix_tokens) // 2
            output_dim = self.gpt_embedding_size * clip_prefix_tokens
            self.clip_project = MLP((input_dim, hidden_dim, output_dim))

    """
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        self.training = mode
        self.clip_project.train(mode)
        self.gpt.eval()
        return self
    """

    def setup_input_output_embeddings(self, bsz, prefix_len):
        self.gpt.resize_token_embeddings(len(self.tokenizer) + (prefix_len * bsz))

    def forward(self, image_feats):
        bsz = image_feats.shape[0]

        prefix_embed = self.clip_project(image_feats)

        prefix_embed = prefix_embed.reshape(bsz, self.clip_prefix_tokens, -1)
        bsz, prefix_len, h_dim = prefix_embed.shape

        bias = torch.Tensor([0.0] * (len(self.tokenizer) + (bsz * prefix_len))).to(
            prefix_embed.device
        )
        self.gpt.get_output_embeddings().bias = nn.Parameter(bias, requires_grad=False)
        self.gpt.get_output_embeddings().bias.data[-(bsz * prefix_len) :] = float(
            "-inf"
        )

        prefix_embed_flat = prefix_embed.view(-1, h_dim)
        start, end = len(self.tokenizer), len(self.tokenizer) + (bsz * prefix_len)
        self.gpt.get_input_embeddings().weight.data[start:end] = prefix_embed_flat
        input_ids = torch.arange(start, end)
        input_ids = input_ids.view(bsz, prefix_len).to(prefix_embed.device)

        # BEAM SEARCH CAPTIONS
        generated_bms = self.gpt.generate(
            input_ids,
            max_length=75,
            num_beams=self.beam_size,
            forced_eos_token_id=self.tokenizer.encode(".")[0],
            output_scores=True,
            return_dict_in_generate=True,
        )
        generated_ids_beam_search = generated_bms.sequences
        captions_beam_search = self.tokenizer.batch_decode(
            generated_ids_beam_search[:, prefix_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        captions_beam_search = [
            caption[: caption.index(".") + 1] for caption in captions_beam_search
        ]

        # GREEDY CAPTIONS
        generated_ids_greedy = self.gpt.generate(
            input_ids,
            max_length=75,
            forced_eos_token_id=self.tokenizer.encode(".")[0],
        )
        captions_greedy = self.tokenizer.batch_decode(
            generated_ids_greedy[:, prefix_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        captions_greedy = [
            caption[: caption.index(".") + 1] for caption in captions_greedy
        ]

        """
        # LOG PROBS
        eos_token = self.tokenizer.encode(".")[0]
        sequence_scores = torch.zeros(bsz).to(input_ids.device)
        is_done = [False] * bsz
        for batch_idx in range(bsz):
            for timestep in range(len(generated_bms.scores)):
                if is_done[batch_idx]:
                    continue

                ts_scores = generated_bms.scores[timestep].view(bsz, self.beam_size, -1)
                beam_idx = generated_bms.beam_indices[batch_idx][timestep]
                try:
                    score = ts_scores[batch_idx, beam_idx]
                except:
                    breakpoint()

                sequence_scores[batch_idx] += score.max()

                if generated_bms.sequences[batch_idx, timestep] == eos_token:
                    is_done[batch_idx] = True

        log_prob = sequence_scores / len(generated_bms.scores)
        """
        log_prob = generated_bms.sequences_scores

        return captions_beam_search, captions_greedy, log_prob


class ClipCapSender(nn.Module):
    def __init__(
        self,
        clip_prefix_tokens: int,
        clip_model: str,
        clip_cap_path: str,
        beam_size: int = 5,
    ):
        super(ClipCapSender, self).__init__()

        self.clip_model = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip_model)
        self.clip_model.eval()

        self.clipcap = ClipCapModel(
            clip_prefix_size=self.clip_model.visual.output_dim,
            clip_prefix_tokens=clip_prefix_tokens,
            beam_size=beam_size,
        )
        if clip_cap_path is not None:
            self.clipcap.load_state_dict(torch.load(clip_cap_path))

    def setup_clipcap(self, batch_size, n_prefix_tokens):
        self.clipcap.setup_input_output_embeddings(batch_size, n_prefix_tokens)

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        image_feats = self.clip_model.encode_image(images)
        caption_beam_search, caption_greedy, log_prob = self.clipcap(image_feats)
        return caption_beam_search, caption_greedy, log_prob
