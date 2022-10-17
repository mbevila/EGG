# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(
        self,
        train_emb_path: str,
        train_nns_path: str,
        test_emb_path: str,
        test_nns_path: str,
        num_hard_negatives: int = 10,
        in_batch_negatives: bool = True,
        test_w_negatives: bool = False,
        num_return_sequences: int = 1,
        logit_scale: float = 1.0,
    ):
        super().__init__()
        assert num_hard_negatives > 0 or num_hard_negatives == -1 or in_batch_negatives

        # TODO make paths to negatives optional
        if num_hard_negatives == -1:
            self.register_buffer('train_emb',
                torch.load(train_emb_path, map_location="cpu"), persistent=False)
            self.register_buffer('test_emb',
                torch.load(test_emb_path, map_location="cpu"), persistent=False)

        else:
            self.train_emb = torch.load(train_emb_path, map_location="cpu")
            self.test_emb = torch.load(test_emb_path, map_location="cpu")

        self.train_nns = torch.load(train_nns_path, map_location="cpu")
        self.test_nns = torch.load(test_nns_path, map_location="cpu")

        self.num_hard_negatives = num_hard_negatives
        self.in_batch_negatives = in_batch_negatives

        self.num_return_sequences = num_return_sequences
        self.logit_scale = logit_scale
        self.test_w_negatives = test_w_negatives

    def get_similarity_scores(self, elem_idxs, text_feats, aux_input=None):
        elem_idxs = elem_idxs.squeeze()

        emb = self.train_emb if self.training else self.test_emb
        nns = self.train_nns if self.training else self.test_nns

        # to disable negatives: set hard negatives to 0 and in_batch_negatives to True
        if self.training or self.test_w_negatives:
            num_hard_negatives = self.num_hard_negatives
            in_batch_negatives = self.in_batch_negatives
        else:
            num_hard_negatives = 0
            in_batch_negatives = True

        if num_hard_negatives == -1:

            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            image_feats_self = emb[nns[elem_idxs][:, :1].long().to(emb.device)]
            image_feats_coll = emb

            cosine_sims_self = self.logit_scale.exp() * torch.einsum(
                "be,bse->bs", text_feats, image_feats_self)
            cosine_sims_coll = self.logit_scale.exp() * torch.einsum(
                "be,ne->bn",  text_feats, image_feats_coll)

            cosine_sims_coll.masked_fill_(
                torch.nn.functional.one_hot(
                    nns[elem_idxs][:, 0].long().to(emb.device),
                    num_classes=image_feats_coll.size(0)
                ).bool(),
                float('-inf')
            )
            cosine_sims = torch.cat([cosine_sims_self, cosine_sims_coll], dim=1)
        else:

            # fetches embeddings of nearest-neighbor hard negatives
            batch_nns = nns[elem_idxs][:, : num_hard_negatives + 1].long()

            # batch x num_negatives + 1 x embed_dim
            image_feats_negatives = emb[batch_nns].to(text_feats.device)

            if self.training:
                image_feats_negatives = image_feats_negatives.repeat_interleave(
                    self.num_return_sequences, 0
                )

            # hard negatives similarity scores
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            cosine_negatives = self.logit_scale.exp() * torch.einsum(
                "be,bne->bn", text_feats, image_feats_negatives
            )

            cosine_sims = cosine_negatives
            # in-batch negatives similarity scores (cat'd to hard negatives if computed)
            if in_batch_negatives:
                # hard negatives set to 0 is equivalent to in-batch negatives
                cosine_in_batch = (
                    self.logit_scale.exp() * text_feats @ image_feats_negatives[:, 0].t()
                )
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
        acc = (cosine_sims.argmax(dim=1) == labels).detach().float()

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
