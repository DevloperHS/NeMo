from typing import Optional, Union, Dict, List

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.asr.models import ASRModel, EncDecCTCModel
from nemo.collections.asr.models.wav2vec.wav2vec_model import Wav2VecEncoderModel
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER


class Wav2VecASRModel(ASRModel, EncDecCTCModel):

    def __init__(self,
                 encoder: Wav2VecEncoderModel,
                 cfg: DictConfig,
                 trainer: Trainer):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus

        ASRModel.__init__(self, cfg=cfg, trainer=trainer)
        self.encoder = encoder
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(num_classes=self.decoder.num_classes_with_blank - 1)

        # Setup metric objects
        self._wer = WER(
            vocabulary=self.decoder.vocabulary,
            batch_dim_index=0,
            use_cer=False,
            ctc_decode=True
        )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    @typecheck()
    def forward(self, input_signal, input_signal_length):
        encoded, encoded_len = self.encoder(
            audio_signal=input_signal,
            length=input_signal_length
        )
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, encoded_len, greedy_predictions