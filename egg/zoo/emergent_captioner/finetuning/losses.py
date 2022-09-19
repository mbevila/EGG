# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(
        self,
        num_hard_negatives=10,
        in_batch_negatives=True,
    ):
        super().__init__()
        assert num_hard_negatives > 0 or in_batch_negatives

        self.emb = torch.load(
            "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.emb.pt",
            map_location="cpu",
        )
        self.nns = torch.load(
            "/private/home/rdessi/EGG/egg/zoo/emergent_captioner/hard_negatives/flickr/train_flickr.nns.pt",
            map_location="cpu",
        )
        self.num_hard_negatives = num_hard_negatives
        self.in_batch_negatives = in_batch_negatives

    def get_similarity_scores(self, elem_idxs, text_feats, aux_input=None):
        elem_idxs = elem_idxs.squeeze()

        # fetches embeddings of nearest-neighbor hard negatives
        batch_nns = self.nns[elem_idxs][:, : self.num_hard_negatives + 1].long()

        # batch x num_negatives + 1 x embed_dim
        image_feats_negatives = self.emb[batch_nns].to(text_feats.device)

        # hard negatives similarity scores
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        cosine_negatives = torch.einsum("bne,be->bn", image_feats_negatives, text_feats)

        cosine_sims = cosine_negatives
        # in-batch negatives similarity scores (cat'd to hard negatives if computed)
        if self.in_batch_negatives:
            # hard negatives set to 0 is equivalent to in-batch negatives
            cosine_in_batch = text_feats @ image_feats_negatives[:, 0].t()
            # diag mask because we don't want to count the current element twice
            cosine_in_batch.fill_diagonal_(float("-inf"))
            cosine_sims = torch.cat([cosine_negatives, cosine_in_batch], dim=1)

        aux_input["receiver_output"] = cosine_sims

        return cosine_sims

    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        aux_input,
    ):
        raise NotImplementedError


class DiscriminativeLoss(Loss):
    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        aux_input,
    ):
        cosine_sims = self.get_similarity_scores(labels, receiver_output, aux_input)

        labels = torch.zeros(receiver_output.shape[0]).long().to(receiver_output.device)

        loss = F.cross_entropy(cosine_sims, labels, reduction="none")
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()

        return loss, {"acc": acc}


class AccuracyLoss(Loss):
    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        aux_input,
    ):
        cosine_sims = self.get_similarity_scores(labels, receiver_output, aux_input)

        labels = torch.zeros(receiver_output.shape[0]).long().to(receiver_output.device)

        acc = (cosine_sims.argmax(dim=1) == labels).detach().float()

        return -acc, {"acc": acc}


class SimilarityLoss(Loss):
    def forward(
        self,
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        aux_input,
    ):
        cosine_sims = self.get_similarity_scores(labels, receiver_output, aux_input)
        # maximising similarity between text and image in the feature space
        return -cosine_sims[0], {}
