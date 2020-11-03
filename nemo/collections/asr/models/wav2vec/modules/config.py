from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from torch import nn


@dataclass
class ConvConfig:
    conv_pos: int = 128
    conv_pos_groups: int = 16


class Wav2VecActivationType(Enum):
    relu = nn.ReLU
    gelu = nn.GELU
    tanh = nn.Tanh


@dataclass
class TransformerSentenceEncoderConfig:
    encoder_layers: int = 12
    encoder_layerdrop: float = 0.05
    embedding_dim: int = 768
    ffn_embedding_dim: int = 3072
    num_attention_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: Wav2VecActivationType = Wav2VecActivationType.relu
    layer_norm_first: bool = False


@dataclass
class TransformerEncoderConfig:
    dropout: float = 0.1
    conv: ConvConfig = ConvConfig()
    encoder: TransformerSentenceEncoderConfig = TransformerSentenceEncoderConfig()


@dataclass
class QuantizeConfig:
    quantize_targets: bool = True
    quantize_input: bool = False
    sample_quantizer: bool = False
    latent_vars: int = 320
    latent_groups: int = 2
    latent_dim: int = 0
    latent_temp: tuple = (2, 0.5, 0.999995)


@dataclass
class ConvFeaturesConfig:
    extractor_mode: str = 'default'
    conv_bias: bool = False
    conv_feature_layers: List = field(
        default_factory=lambda: [(512, 10, 5), (512, 8, 4)] + [(512, 4, 2)] * 3 + [(512, 1, 1)]
    )  # Default conv layers as defined in the fairseq repo


@dataclass
class LossConfig:
    infonce: bool = True
    loss_weights: Optional[tuple] = (0.1, 10)


@dataclass
class Wav2VecEncoderModelConfig:
    loss: LossConfig = LossConfig()
    quantize: QuantizeConfig = QuantizeConfig()
    conv_features: ConvFeaturesConfig = ConvFeaturesConfig()
    transformer_encoder: TransformerEncoderConfig = TransformerEncoderConfig()

    mask_prob: float = 0.65
    mask_selection: str = 'static'
    mask_other: int = 0
    mask_length: int = 10
    no_mask_overlap: bool = False
    mask_min_space: int = 1

    mask_channel_prob: float = 0
    mask_channel_selection: str = 'static'
    mask_channel_other: int = 0
    mask_channel_length: int = 10
    no_mask_channel_overlap: bool = False
    mask_channel_min_space: int = 1

    dropout_input: float = 0.1
    dropout_features: float = 0.1

    final_dim: int = 0

    n_negatives: int = 100
    cross_sample_negatives: int = 0
    codebook_negatives: int = 0
    negatives_from_everywhere: bool = False

    logit_temp: float = 0.1

    target_glu: bool = False

    feature_grad_mult: float = 0.1
