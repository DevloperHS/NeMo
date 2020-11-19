import logging

import torch
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset, TarredAudioToCharDataset
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.core import ModelPT
from omegaconf import DictConfig
from pytorch_lightning import Trainer


class Wav2VecBase(ModelPT):
    def __init__(self, pretraining: bool, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 0
        self.local_rank = 0
        self.pretraining = pretraining
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        super().__init__(cfg=cfg, trainer=trainer)

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
                labels=[] if self.pretraining else config['labels'],
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
                return_pad_mask=True,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = AudioToCharDataset(
                manifest_filepath=config['manifest_filepath'],
                labels=[] if self.pretraining else config['labels'],
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                max_utts=config.get('max_utts', 0),
                return_pad_mask=True,
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: DictConfig):
        self._update_dataset_config(dataset_name='train', config=train_data_config)
        self._train_dl = self.setup_dataloader(train_data_config)

    def setup_validation_data(self, val_data_config: DictConfig):
        self._update_dataset_config(dataset_name='validation', config=val_data_config)
        self._validation_dl = self.setup_dataloader(val_data_config)

    def setup_test_data(self, test_data_config: DictConfig):
        self._update_dataset_config(dataset_name='test', config=test_data_config)
        self._test_dl = self.setup_dataloader(test_data_config)
