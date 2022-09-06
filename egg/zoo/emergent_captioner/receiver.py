# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch.nn as nn

import clip

from egg.zoo.emergent_captioner.utils import convert_models_to_fp32


class ClipReceiver(nn.Module):
    def __init__(
        self,
        clip_model: str,
        return_embeddings: bool = False,
    ):
        super(ClipReceiver, self).__init__()
        self.clip = clip.load(clip_model)[0]
        convert_models_to_fp32(self.clip)
        self.return_embeddings = return_embeddings
        self.clip.eval()

    def forward(self, message, images, aux_input=None):
        text = clip.tokenize(message, truncate=True).to(images.device)
        if self.return_embeddings:
            return self.encode_text(text)
        else:
            _, clip_logits = self.clip(images, text)
            return clip_logits
