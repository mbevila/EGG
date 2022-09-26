# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

import clip

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class ClipReceiver(nn.Module):
    def __init__(self, clip_model: str, num_trajectories: int = 1):
        super(ClipReceiver, self).__init__()
        self.clip = clip.load(clip_model)[0]
        self.clip.visual = None  # detaching clip ViT to save memory
        convert_models_to_fp32(self.clip)
        self.clip.eval()

        self.num_trajectories = num_trajectories

    def encode_captions(self, text):
        x = self.clip.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.clip.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return x

    def forward(self, message, images, aux_input=None):
        text = clip.tokenize(message, truncate=True).to(images.device)
        # _, clip_logits = self.clip(images, text)
        # return clip_logits
        with torch.no_grad():
            text_feats = self.encode_captions(text)
        return text_feats
