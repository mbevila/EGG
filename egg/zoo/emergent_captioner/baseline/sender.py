# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Tuple

import clip
from timebudget import timebudget
import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    LogitsProcessor,
    StoppingCriteria,
    # StoppingCriteriaList,
)

from egg.zoo.emergent_captioner.baseline.utils import convert_models_to_fp32


class StopTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: int):
        super(StopTokenCriteria, self).__init__()
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs
    ) -> bool:
        # all beams have produced the stop token which is the pad token
        return torch.all((input_ids == self.eos_token_id)[:, -1])


class StopTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, stop_tokens_idx, eos_token_id, tokenizer):
        self.stop_tokens_idx = stop_tokens_idx
        self.eos_token_id = eos_token_id

        self.tokenizer = tokenizer
        self.is_done = None
        # self.idx_to_check = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.is_done is None:
            self.is_done = torch.zeros(scores.shape[0]).to(scores.device)

        # if self.is_done.bool().any().item():
        #    breakpoint()

        unfinished = (self.is_done == 0).nonzero(as_tuple=True)
        to_check = input_ids[unfinished]
        for i, input_id in enumerate(to_check):
            if input_id[-1].item() in self.stop_tokens_idx:
                beam_idx = unfinished[0][i]
                scores[beam_idx] = -float("inf")
                scores[beam_idx, self.eos_token_id] = 0.0
                self.is_done[beam_idx] = 1

        """
        if self.idx_to_check is None:
            self.idx_to_check = set((_ for _ in range(input_ids.shape[0])))

        with timebudget("find stopped tokens"):
            is_done = torch.zeros(scores.shape[0]).to(scores.device).bool()
            for idx in self.idx_to_check.copy():
                if input_ids[idx, -1] in self.stop_tokens_idx:
                    is_done[idx] = True
                    self.idx_to_check.remove(idx)

        with timebudget("set tokens"):
            if any(is_done):
                scores[is_done] = -float("inf")
                scores[is_done, self.eos_token_id] = 0
        """

        return scores


class HumanCaptionSender(nn.Module):
    def forward(self, x, aux_input=None):
        return aux_input["caption"]


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


class ClipCaptionModel(nn.Module):
    def __init__(
        self,
        clip_prefix_size: int,
        clip_prefix_tokens: int,
        use_beam_search: bool = True,
        beam_size: int = 5,
    ):
        super(ClipCaptionModel, self).__init__()

        self.clip_prefix_tokens = clip_prefix_tokens

        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.gpt.config.pad_token_id = self.gpt.config.eos_token_id

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.gpt.config.pad_token_id

        self.use_beam_search = use_beam_search
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

        stop_word_ids = []  # [self.tokenizer.pad_token_id]
        for idx in range(len(self.tokenizer)):
            x = self.tokenizer.convert_ids_to_tokens(idx)
            if "." in x:  # x.endswith(".") or x.startswith("."):
                stop_word_ids.append(x)

        stop_word_ids = list(set([x for xs in stop_word_ids for x in xs]))
        self.stop_word_ids = self.tokenizer(
            stop_word_ids, add_prefix_space=True, add_special_tokens=False
        ).input_ids
        """
        self.logits_processor = StopTokenLogitsProcessor(
            self.stop_idxs, self.gpt.config.eos_token_id, self.tokenizer
        )
        """

    def setup_input_output_embeddings(self, bsz, prefix_len):
        self.gpt.resize_token_embeddings(len(self.tokenizer) + (prefix_len * bsz))

    def forward(self, image_feats):
        bsz = image_feats.shape[0]

        with torch.no_grad():
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

        if self.use_beam_search:
            generated_ids = self.gpt.generate(
                input_ids, max_length=75, num_beams=self.beam_size
            )
        else:
            generated_ids = self.gpt.generate(input_ids, max_length=75)

        captions = self.tokenizer.batch_decode(
            generated_ids[:, prefix_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        captions = [caption[: caption.index(".") + 1] for caption in captions]
        return captions


class ClipCapSender(nn.Module):
    def __init__(
        self,
        clip_prefix_tokens: int,
        clip_model: str,
        clip_cap_path: str,
        use_beam_search: bool = True,
        beam_size: int = 5,
    ):
        super(ClipCapSender, self).__init__()

        self.clip_model = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip_model)
        self.clip_model.eval()

        self.clipcap = ClipCaptionModel(
            clip_prefix_size=self.clip_model.visual.output_dim,
            clip_prefix_tokens=clip_prefix_tokens,
            use_beam_search=use_beam_search,
            beam_size=beam_size,
        )
        if clip_cap_path is not None:
            self.clipcap.load_state_dict(torch.load(clip_cap_path))

    def setup_clipcap(self, batch_size, n_prefix_tokens):
        self.clipcap.setup_input_output_embeddings(batch_size, n_prefix_tokens)

    def forward(self, images: torch.Tensor, aux_input: Dict[Any, torch.Tensor] = None):
        with torch.no_grad():
            image_feats = self.clip_model.encode_image(images)
            caption = self.clipcap(image_feats)

        return caption
