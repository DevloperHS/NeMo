import logging
import math
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn

from nemo.collections.asr.data.audio_to_text import TarredAudioToCharDataset, AudioToCharDataset
from nemo.collections.asr.losses.wav2vecloss import Wav2vecCriterion
from nemo.collections.asr.models.wav2vec.modules.config import TransformerSentenceEncoderConfig, \
    TransformerEncoderConfig, QuantizeConfig, ConvFeaturesConfig, Wav2VecEncoderModelConfig
from nemo.collections.asr.models.wav2vec.modules.gumbel_vector_quantizer import GumbelVectorQuantizer
from nemo.collections.asr.models.wav2vec.modules.multihead_attention import MultiheadAttention
from nemo.collections.asr.models.wav2vec.modules.norm import Fp32LayerNorm, Fp32GroupNorm
from nemo.collections.asr.models.wav2vec.modules.utils import compute_mask_indices
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo


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


class Wav2VecEncoderModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
        super().__init__(cfg=cfg, trainer=trainer)

        transformer_encoder_cfg = TransformerEncoderConfig(**cfg.get('transformer_encoder', {}))
        quantize_cfg = QuantizeConfig(**cfg.get('quantize', {}))
        conv_cfg = ConvFeaturesConfig(**cfg.get('conv_features', {}))
        encoder_cfg = Wav2VecEncoderModelConfig(**cfg.get('encoder', {}))

        feature_enc_layers = conv_cfg.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=conv_cfg.extractor_mode,
            conv_bias=conv_cfg.conv_bias
        )

        encoder_embed_dim = transformer_encoder_cfg.encoder.embedding_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim and not quantize_cfg.quantize_input
            else None
        )

        self.mask_prob = encoder_cfg.mask_prob
        self.mask_selection = encoder_cfg.mask_selection
        self.mask_other = encoder_cfg.mask_other
        self.mask_length = encoder_cfg.mask_length
        self.no_mask_overlap = encoder_cfg.no_mask_overlap
        self.mask_min_space = encoder_cfg.mask_min_space

        self.mask_channel_prob = encoder_cfg.mask_channel_prob
        self.mask_channel_selection = encoder_cfg.mask_channel_selection
        self.mask_channel_other = encoder_cfg.mask_channel_other
        self.mask_channel_length = encoder_cfg.mask_channel_length
        self.no_mask_channel_overlap = encoder_cfg.no_mask_channel_overlap
        self.mask_channel_min_space = encoder_cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(encoder_cfg.dropout_input)
        self.dropout_features = nn.Dropout(encoder_cfg.dropout_features)

        self.feature_grad_mult = encoder_cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = encoder_cfg.n_negatives
        self.cross_sample_negatives = encoder_cfg.cross_sample_negatives
        self.codebook_negatives = encoder_cfg.codebook_negatives
        self.negatives_from_everywhere = encoder_cfg.negatives_from_everywhere

        self.logit_temp = encoder_cfg.logit_temp

        final_dim = encoder_cfg.final_dim if encoder_cfg.final_dim > 0 else encoder_embed_dim

        if quantize_cfg.quantize_targets:
            vq_dim = quantize_cfg.latent_dim if quantize_cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=quantize_cfg.latent_vars,
                temp=quantize_cfg.latent_temp,
                groups=quantize_cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if quantize_cfg.quantize_input:
            if quantize_cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = (
                    quantize_cfg.latent_dim if quantize_cfg.latent_dim > 0 else encoder_embed_dim
                )
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=quantize_cfg.latent_vars,
                    temp=quantize_cfg.latent_temp,
                    groups=quantize_cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(transformer_encoder_cfg)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if encoder_cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
        self.loss = Wav2vecCriterion()

    def setup_dataloader(self, config: DictConfig):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                    'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size'])
            dataset = TarredAudioToCharDataset(
                audio_tar_filepaths=config['tarred_audio_filepaths'],
                manifest_filepath=config['manifest_filepath'],
                labels=[],
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                max_utts=config.get('max_utts', 0),
                trim=config.get('trim_silence', False),
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = AudioToCharDataset(
                manifest_filepath=config['manifest_filepath'],
                labels=[],
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                max_utts=config.get('max_utts', 0),
                blank_index=config.get('blank_index', -1)
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
        )

    def setup_training_data(self, train_data_config: DictConfig):
        self._train_dl = self.setup_dataloader(train_data_config)

    def setup_validation_data(self, val_data_config: DictConfig):
        self._validation_dl = self.setup_dataloader(val_data_config)

    def setup_test_data(self, test_data_config: DictConfig):
        self._test_dl = self.setup_dataloader(test_data_config)

    def training_step(self, batch, batch_ix):
        loss, sample_size, logging_output = self.loss(
            model=self,
            sample=batch
        )
        logs = {
            'train_loss': loss,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
        }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, sample_size, logging_output = self.loss(
            model=self,
            sample=batch
        )
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, sample_size, logging_output = self.loss(
            model=self,
            sample=batch
        )
        return {'test_loss': loss}

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'validation_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': logs}

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': val_loss_mean}
        return {'test_loss': val_loss_mean, 'log': logs}

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb.type_as(x)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
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
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
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
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def forward(self, source, padding_mask=None, mask=True, features_only=False):

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

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

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, net_output):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

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


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerEncoderConfig):
        super().__init__()

        conv_cfg = cfg.conv

        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder.embedding_dim

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
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    cfg=encoder_cfg,
                    dropout=self.dropout,
                    embedding_dim=self.embedding_dim
                )
                for _ in range(encoder_cfg.encoder_layers)
            ]
        )

        self.layer_norm_first = encoder_cfg.layer_norm_first
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layerdrop = cfg.encoder.encoder_layerdrop
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

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            cfg: TransformerSentenceEncoderConfig,
            embedding_dim: float = 768,
            dropout: float = 0.1,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = cfg.activation_dropout

        # Initialize blocks
        self.activation_fn = cfg.activation_fn.value()
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            cfg.num_attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = cfg.layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, cfg.ffn_embedding_dim)
        self.fc2 = nn.Linear(cfg.ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
