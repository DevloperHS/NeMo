#requires sox with mp3 support
#For example sudo apt-get install libsox-fmt-mp3

import argparse
import csv
import json
import logging
import multiprocessing
import os
import tarfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

import sox
from sox import Transformer
import wget
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Downloads and processes Mozilla Common Voice dataset.')
parser.add_argument("--data_root", default='./', type=str, help="Directory to store the dataset.")
parser.add_argument('--manifest_dir', default='manifest_dir/', type=str, help='Output directory for manifests')
parser.add_argument("--num_workers", default=multiprocessing.cpu_count(), type=int, help="Workers to process dataset.")
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--files_to_process', nargs='+', default=['test.tsv', 'dev.tsv', 'train.tsv'],
                    type=str, help='list of *.csv file names to process')
parser.add_argument('--version', default='cv-corpus-7.0-2021-07-21',
                    type=str, help='Version of the dataset (obtainable via https://commonvoice.mozilla.org/en/datasets')
parser.add_argument('--language', default='hi',
                    type=str, help='Which language to download.(default english,'
                                   'check https://commonvoice.mozilla.org/en/datasets for more language codes')
args = parser.parse_args()


# https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-7.0-2021-07-21/cv-corpus-7.0-2021-07-21-en.tar.gz
COMMON_VOICE_URL = f"https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/" \
                   "{}/{}-{}.tar.gz".format(args.version, args.version, args.language)


def create_manifest(
        data: List[tuple],
        output_name: str,
        manifest_path: str):
    output_file = Path(manifest_path) / output_name
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with output_file.open(mode='w') as f:
        for wav_path, duration, text in tqdm(data, total=len(data)):
            f.write(
                json.dumps({
                    'audio_filepath': wav_path,
                    "duration": duration,
                    'text': text
                }) + '\n'
            )


def process_files(csv_file, data_root, num_workers):
    """ Read *.csv file description, convert mp3 to wav, process text.
        Save results to data_root.
    Args:
        csv_file: str, path to *.csv file with data description, usually start from 'cv-'
        data_root: str, path to dir to save results; wav/ dir will be created
    """
    wav_dir = os.path.join(data_root, 'wav/')
    os.makedirs(wav_dir, exist_ok=True)
    audio_clips_path = os.path.dirname(csv_file) + '/clips/'

    def process(x):
        file_path, text = x
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        text = text.lower().strip()
        audio_path = os.path.join(audio_clips_path, file_path)
        output_wav_path = os.path.join(wav_dir, file_name + '.wav')

        tfm = Transformer()
        tfm.rate(samplerate=args.sample_rate)
        tfm.build(
            input_filepath=audio_path,
            output_filepath=output_wav_path
        )
        duration = sox.file_info.duration(output_wav_path)
        return output_wav_path, duration, text

    logging.info('Converting mp3 to wav for {}.'.format(csv_file))
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        next(reader, None)  # skip the headers
        data = [(row['path'], row['sentence']) for row in reader]
        with ThreadPool(num_workers) as pool:
            data = list(tqdm(pool.imap(process, data), total=len(data)))
    return data


def main():
    data_root = args.data_root
    os.makedirs(data_root, exist_ok=True)

    target_unpacked_dir = os.path.join(data_root, "CV_unpacked")

    if os.path.exists(target_unpacked_dir):
        logging.info('Find existing folder {}'.format(target_unpacked_dir))
    else:
        logging.info("Could not find Common Voice, Downloading corpus...")

        filename = wget.download(COMMON_VOICE_URL, data_root)
        target_file = os.path.join(data_root, os.path.basename(filename))

        os.makedirs(target_unpacked_dir, exist_ok=True)
        logging.info("Unpacking corpus to {} ...".format(target_unpacked_dir))
        tar = tarfile.open(target_file)
        tar.extractall(target_unpacked_dir)
        tar.close()

    folder_path = os.path.join(target_unpacked_dir, args.version + f'/{args.language}/')

    for csv_file in args.files_to_process:
        data = process_files(
            csv_file=os.path.join(folder_path, csv_file),
            data_root=os.path.join(data_root, os.path.splitext(csv_file)[0]),
            num_workers=args.num_workers
        )
        logging.info('Creating manifests...')
        create_manifest(
            data=data,
            output_name=f'commonvoice_{os.path.splitext(csv_file)[0]}_manifest.json',
            manifest_path=args.manifest_dir,
        )


if __name__ == "__main__":
    main()
