from dataclasses import dataclass, field, MISSING
from enum import Enum
from typing import Optional, List, Any

from omegaconf import DictConfig
from torch import nn

from nemo.core.config import PolynomialDecayAnnealingParams


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
    encoder_layerdrop: float = 0.0
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
    quantize_targets: bool = False
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
        default_factory=lambda:
        [
            (512, 10, 5),
            (512, 8, 4)
        ] +
        [
            (512, 4, 2)
        ] * 3 +
        [
            (512, 1, 1)
        ]
    )  # Default conv layers as defined in the fairseq repo


@dataclass
class DataConfig:
    manifest_path: str = ''
    sample_rate: int = 16000
    max_sample_size: Optional[int] = field(default_factory=lambda: None)
    min_sample_size: Optional[int] = field(default_factory=lambda: None)
    min_length: Optional[int] = field(default_factory=lambda: None)
    pad: bool = False
    normalize: bool = False
    batch_size: int = 32
    drop_last: bool = False
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class Wav2VecModelConfig:
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

    dropout_input: float = 0
    dropout_features: float = 0

    final_dim: int = 0

    n_negatives: int = 100
    cross_sample_negatives: int = 0
    codebook_negatives: int = 0
    negatives_from_everywhere: bool = False

    logit_temp: float = 0.1

    target_glu: bool = False

    # quantize
    quantize: QuantizeConfig = QuantizeConfig()

    feature_grad_mult: float = 1.0

    # conv_features
    conv_features: ConvFeaturesConfig = ConvFeaturesConfig()
