import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from nemo.collections.asr.losses.wav2vecloss import Wav2VecCriterion
from nemo.collections.asr.models.wav2vec.modules.config import (
    Wav2VecTransformerConfig,
    Wav2VecEncoderModelConfig,
)
from nemo.collections.asr.models.wav2vec.modules.gumbel_vector_quantizer import GumbelVectorQuantizer
from nemo.collections.asr.models.wav2vec.modules.utils import compute_mask_indices, init_bert_params, TransposeLast, \
    SamePad
from nemo.collections.asr.models.wav2vec.wav2vec_base import Wav2VecBase
from nemo.core.classes.common import PretrainedModelInfo
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


@dataclass
class Wav2VecEncoderOutput:
    """
    Helper class for storing all the outputs from the Encoder model.
    """
    logits: torch.tensor
    targets: torch.tensor
    sampled_negatives: torch.tensor
    padding_mask: torch.tensor
    features_penalty: torch.tensor
    probs_ppl: torch.tensor
    cur_codebook_temp: float


class Wav2VecEncoderModel(Wav2VecBase):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(pretraining=True, cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Wav2VecEncoderModelConfig)
        cfg = cfg.get('params', {})
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.merge(schema, cfg)

        feature_enc_layers = cfg.conv_features.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            mode=cfg.conv_features.extractor_mode,
            conv_bias=cfg.conv_features.conv_bias,
        )

        encoder_embed_dim = cfg.transformer_encoder.encoder.embedding_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim and not cfg.quantize.quantize_input
            else None
        )

        self.mask_cfg = cfg.mask

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        final_dim = cfg.final_dim if cfg.final_dim > 0 else encoder_embed_dim
        self.final_dim = final_dim
        if cfg.quantize.quantize_targets:
            vq_dim = cfg.quantize.latent_dim if cfg.quantize.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.quantize.latent_vars,
                temp=cfg.quantize.latent_temp,
                groups=cfg.quantize.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize.quantize_input:
            if cfg.quantize.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.quantize.latent_dim if cfg.quantize.latent_dim > 0 else encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.quantize.latent_vars,
                    temp=cfg.quantize.latent_temp,
                    groups=cfg.quantize.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, encoder_embed_dim)

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_embed_dim).uniform_())

        self.encoder = Wav2VecTransformerEncoder(cfg.transformer_encoder)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
        self.loss = Wav2VecCriterion(
            feature_loss_weight=cfg.loss.feature_loss_weight,
            prob_ppl_weight=cfg.loss.prob_ppl_weight,
            logit_temp=cfg.logit_temp
        )

    def training_step(self, batch, batch_idx):
        loss, feature_loss, prob_ppl_loss = self._step(batch)

        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('loss', loss)
        self.log('feature_loss', feature_loss)
        self.log('prob_ppl_loss', prob_ppl_loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, feature_loss, prob_ppl_loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, feature_loss, prob_ppl_loss = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)

    def _step(self, batch):
        audio_signal, audio_lengths, _, _, padding_mask = batch
        self._update_quantizer_temp()
        model_output = self(audio_signal, padding_mask)
        loss, feature_loss, prob_ppl_loss = self.loss(
            logits=model_output.logits,
            targets=model_output.targets,
            negatives=model_output.sampled_negatives,
            prob_ppl_loss=model_output.probs_ppl,
            feature_loss=model_output.features_penalty
        )
        return loss, feature_loss, prob_ppl_loss

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def _update_quantizer_temp(self):
        if self.quantizer:
            self.quantizer.set_num_updates(self.trainer.global_step)
        if self.input_quantizer:
            self.input_quantizer.set_num_updates(self.trainer.global_step)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_cfg.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_cfg.mask_prob,
                self.mask_cfg.mask_length,
                self.mask_cfg.mask_type,
                self.mask_cfg.mask_other,
                min_masks=2,
                no_overlap=self.mask_cfg.no_mask_overlap,
                min_space=self.mask_cfg.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb.type_as(x)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_cfg.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_cfg.mask_channel_prob,
                self.mask_cfg.mask_channel_length,
                self.mask_cfg.mask_channel_type,
                self.mask_cfg.mask_channel_other,
                no_overlap=self.mask_cfg.no_mask_channel_overlap,
                min_space=self.mask_cfg.mask_channel_min_space,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()

                cross_neg_idxs = torch.randint(
                    low=0, high=cross_high - 1, size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def forward(self, source, padding_mask=None, mask=True, features_only=False) -> Union[tuple, Wav2VecEncoderOutput]:
        prob_ppl, cur_codebook_temp = None, None

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_penalty = features.float().pow(2).mean()  # L2 Norm on features

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        if self.input_quantizer:
            features, prob_ppl, cur_codebook_temp = self.input_quantizer(features)
            features = self.project_inp(features)
        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)

        if features_only:
            return x, padding_mask

        if self.quantizer:
            y, prob_ppl, cur_codebook_temp = self.quantizer(y)
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features)
                sampled_negatives, _ = self.sample_negatives(neg_cands, y.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                sampled_negatives, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(y.size(0) * y.size(1), self.codebook_negatives)
                cb_negs = cb_negs.view(self.codebook_negatives, y.size(0), y.size(1), -1)  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                sampled_negatives = torch.cat([sampled_negatives, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                sampled_negatives, _ = self.sample_negatives(unmasked_features, y.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                sampled_negatives, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            sampled_negatives = self.target_glu(sampled_negatives)

        x = self.final_proj(x)
        output = Wav2VecEncoderOutput(
            logits=x,
            targets=y,
            sampled_negatives=sampled_negatives,
            padding_mask=padding_mask,
            features_penalty=features_penalty,
            probs_ppl=prob_ppl,
            cur_codebook_temp=cur_codebook_temp
        )
        return output

    def extract_features(self, source, padding_mask, mask=False):
        return self.forward(source, padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            mode: str = "default",
            conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class Wav2VecTransformerEncoder(nn.Module):
    def __init__(self, cfg: Wav2VecTransformerConfig):
        super().__init__()

        conv_cfg = cfg.conv

        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder.embedding_dim
        self.layer_norm_first = cfg.encoder.layer_norm_first

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_cfg.conv_pos,
            padding=conv_cfg.conv_pos // 2,
            groups=conv_cfg.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_cfg.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_cfg.conv_pos), nn.GELU())

        encoder_cfg = cfg.encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=encoder_cfg.num_attention_heads,
                dim_feedforward=encoder_cfg.ffn_embedding_dim,
                dropout=self.dropout,
                activation=encoder_cfg.activation_fn.value
            ),
            num_layers=encoder_cfg.encoder_layers
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x
