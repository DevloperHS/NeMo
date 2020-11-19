# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


class Wav2VecCriterion(torch.nn.Module):
    def __init__(self, feature_loss_weight, prob_ppl_weight, logit_temp):
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.prob_ppl_weight = prob_ppl_weight
        self.logit_temp = logit_temp

    def forward(self,
                logits: torch.tensor,
                targets: torch.tensor,
                negatives: torch.tensor,
                feature_loss: torch.tensor,
                prob_ppl_loss: torch.tensor,
                reduce=True) -> [torch.tensor, torch.tensor, torch.tensor]:
        """
        Compute the contrastive loss with respect to the model outputs and sampled negatives from quantizer codebooks.
        Args:
            logits: Model activations
            targets: The true target quantized representation
            negatives: Sampled negatives from the quantizer codebooks. Sampled from all other timesteps.
            feature_loss: Feature penalty (L2 Norm)
            prob_ppl_loss:
            reduce: Reduce loss via sum reduction (Default true)
        Returns:
            output loss values, feature loss, prob_ppl loss (after scaling).
        """

        # Calculate similarity between logits and all targets
        neg_is_pos = (targets == negatives).all(-1)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0)
        logits = torch.cosine_similarity(logits.float(), targets.float(), dim=-1).type_as(logits)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        logits = logits.transpose(0, 2)  # TODO BxTxD ->
        logits = logits.reshape(-1, logits.size(-1))  # TODO BxTxD ->

        # Create similarity targets
        targets = logits.new_zeros(logits.size(1) * logits.size(2), dtype=torch.long)

        loss = F.cross_entropy(logits, targets, reduction="sum" if reduce else "none")

        sample_size = targets.numel()

        if self.feature_loss_weight != 0:
            feature_loss = self.feature_loss_weight * feature_loss.float() * sample_size
            loss += feature_loss

        if self.prob_ppl_weight != 0:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss.float() * sample_size
            loss += prob_ppl_loss

        return loss, feature_loss, prob_ppl_loss
