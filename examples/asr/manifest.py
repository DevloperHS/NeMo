#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import json
import os

import soundfile
from sklearn.model_selection import train_test_split


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', metavar='DIR', help='root directory containing flac files to index')
    parser.add_argument('--valid-percent', default=0.01, type=float, metavar='D',
                        help='percentage of data to use as validation set (between 0 and 1)')
    parser.add_argument('--dest', default='.', type=str, metavar='DIR', help='output directory')
    parser.add_argument('--ext', default='wav', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed')
    parser.add_argument('--path-must-contain', default=None, type=str, metavar='FRAG',
                        help='if set, path must contain this substring for a file to be included in the manifest')
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, '**/*.' + args.ext)

    files = [
        {
            'audio_filepath': os.path.realpath(path),
            'duration': soundfile.info(path).duration,
            'text': ''
        } for path in glob.iglob(search_path, recursive=True)
    ]

    train_files, val_files = train_test_split(
        files,
        test_size=args.valid_percent,
        random_state=args.seed
    )

    with open(os.path.join(args.dest, 'train.json'), 'w') as train_f:
        train_f.write('\n'.join(json.dumps(x) for x in train_files))

    with open(os.path.join(args.dest, 'valid.json'), 'w') as valid_f:
        valid_f.write('\n'.join(json.dumps(x) for x in val_files))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
