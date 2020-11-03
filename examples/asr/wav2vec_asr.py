# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from omegaconf import DictConfig

from nemo.collections.asr.models.wav2vec.wav2vec_asr_model import Wav2VecASRModel
from nemo.collections.asr.models.wav2vec.wav2vec_model import Wav2VecEncoderModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


"""

python examples/asr/wav2vec.py \
        model.train_ds.manifest_path="./examples/asr/train.tsv" \
        model.validation_ds.manifest_path="./examples/asr/valid.tsv" \
        model.test_ds.manifest_path="./examples/asr/valid.tsv" \
        hydra.run.dir="." \
        trainer.gpus=0 \
        trainer.max_epochs=50
        
        
Basic run (on CPU for 50 epochs):
    python examples/asr/wav2vec.py \
        model.train_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_train.json" \
        model.validation_ds.manifest_filepath="/Users/okuchaiev/Data/an4_dataset/an4_val.json" \
        hydra.run.dir="." \
        trainer.gpus=0 \
        trainer.max_epochs=50


Add PyTorch Lightning Trainer arguments from CLI:
    python speech_to_text.py \
        ... \
        +trainer.fast_dev_run=true

Hydra logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/.hydra)"
PTL logs will be found in "$(./outputs/$(date +"%y-%m-%d")/$(date +"%H-%M-%S")/lightning_logs)"

Override some args of optimizer:
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.args.params.betas=[0.8,0.5] \
    model.optim.args.params.weight_decay=0.0001

Overide optimizer entirely
    python speech_to_text.py \
    model.train_ds.manifest_filepath="./an4/train_manifest.json" \
    model.validation_ds.manifest_filepath="./an4/test_manifest.json" \
    hydra.run.dir="." \
    trainer.gpus=2 \
    trainer.max_epochs=2 \
    model.optim.name=adamw \
    model.optim.lr=0.001 \
    ~model.optim.args \
    +model.optim.args.betas=[0.8,0.5]\
    +model.optim.args.weight_decay=0.0005

"""


@hydra_runner(config_path="conf", config_name="wav2vec_asr")
def main(cfg: DictConfig):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    encoder_model = Wav2VecEncoderModel.load_from_checkpoint(cfg.model.encoder_path)
    wav2vec_model = Wav2VecASRModel(encoder=encoder_model, cfg=cfg.model, trainer=trainer)
    trainer.fit(wav2vec_model)
