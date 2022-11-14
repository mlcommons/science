#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# data_loader.py

# SciML-Bench
# Copyright © 2022 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

from pathlib import Path
import h5py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Union, List


# Convert images 1200x1500x9 to 256x256x9 to be used with U-net.
class SLSTRDataLoader:

    def __init__(self,
                 args: dict,
                 paths: Union[Path, List[Path]],
                 shuffle: bool = True,
                 batch_size: int = 32,
                 single_image: bool = False,
                 crop_size: int = 20,
                 no_cache=False):

        if isinstance(paths, Path):
            self._image_paths = Path(paths).glob('**/S3A*.hdf')
        else:
            self._image_paths = paths

        self._image_paths = list(map(str, self._image_paths))
        self._shuffle = shuffle if not single_image else False
        self.single_image = single_image
        self.patch_padding = 'valid' if not single_image else 'same'

        # Parameters from cloudMaskConfig.yaml
        self.patch_size = args['PATCH_SIZE']
        self.n_channels = args['N_CHANNELS']
        self.image_h = args['IMAGE_H']
        self.image_w = args['IMAGE_W']
        self.batch_size = args['batch_size']
        self.no_cache = args['no_cache']
        self.crop_size = args['CROP_SIZE']

        assert len(self._image_paths) > 0, 'No image data found in path!'

    @property
    def input_size(self):
        return self.patch_size, self.patch_size, self.n_channels

    @property
    def output_size(self):
        return (self.patch_size, self.patch_size, 1)

    def _load_data(self, path):
        path = path.decode()

        with h5py.File(path, 'r') as handle:
            refs = handle['refs'][:]
            bts = handle['bts'][:]
            msk = handle['bayes'][:]

        bts = (bts - 270.0) / 22.0
        refs = refs - 0.5
        img = np.concatenate([refs, bts], axis=-1)

        msk[msk > 0] = 1
        msk[msk == 0] = 0
        msk = msk.astype(np.float)

        yield img, msk, path.encode('utf-8')

    def _preprocess_images(self, img, msk, path):
        # Crop & convert to patches
        img = self._transform_image(img)
        msk = self._transform_image(msk)

        if self.single_image:
            return img, path
        else:
            return img, msk

    def _transform_image(self, img):
        # Crop to image which is divisible by the patch size
        # This also removes image boarders which are all zero
        IMAGE_H = self.image_h
        IMAGE_W = self.image_w
        PATCH_SIZE = self.patch_size
        offset_h = (IMAGE_H % PATCH_SIZE) // 2
        offset_w = (IMAGE_W % PATCH_SIZE) // 2

        target_h = IMAGE_H - offset_h * 2
        target_w = IMAGE_W - offset_w * 2

        if not self.single_image:
            img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, target_h, target_w)

        # Covert image from IMAGE_H x IMAGE_W to PATCH_SIZE x PATCH_SIZE
        dims = [1, PATCH_SIZE, PATCH_SIZE, 1]
        strides = dims.copy()

        if self.single_image:
            strides[1] -= self.crop_size
            strides[2] -= self.crop_size

        img = tf.expand_dims(img, axis=0)
        b, h, w, c = img.shape
        img = tf.image.extract_patches(img, dims, strides, [1, 1, 1, 1], padding=self.patch_padding.upper())

        if not self.single_image:
            n, nx, ny, np = img.shape
            img = tf.reshape(img, (n * nx * ny, PATCH_SIZE, PATCH_SIZE, c))

        return img

    def _generator(self, path):
        IMAGE_H = self.image_h
        IMAGE_W = self.image_w
        N_CHANNELS = self.n_channels
        types = (tf.float32, tf.float32, tf.string)
        shapes = (tf.TensorShape([IMAGE_H, IMAGE_W, N_CHANNELS]),
                  tf.TensorShape([IMAGE_H, IMAGE_W, 1]),
                  tf.TensorShape([]))
        dataset = tf.data.Dataset.from_generator(self._load_data,
                                                 output_types=types,
                                                 output_shapes=shapes,
                                                 args=(path,))
        return dataset

    def to_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self._image_paths)

        if self._shuffle:
            dataset = dataset.shuffle(len(self._image_paths))

        dataset = dataset.interleave(self._generator, cycle_length=2, num_parallel_calls=2)
        dataset = dataset.map(self._preprocess_images, num_parallel_calls=2)

        if self.single_image:
            return dataset

        dataset = dataset.unbatch()
        if not self.no_cache:
            dataset = dataset.cache()
        dataset = dataset.prefetch(self.batch_size)

        if self._shuffle:
            dataset = dataset.shuffle(self.batch_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset


# Dataloader specific to this benchmark
def load_datasets(dataset_dir: Path, args: dict):
    data_paths = list(Path(dataset_dir).glob('**/S3A*.hdf'))

    train_paths, test_paths = train_test_split(data_paths, train_size=args['train_split'], random_state=42)

    train_data_loader = SLSTRDataLoader(args, train_paths, batch_size=args['batch_size'], no_cache=args['no_cache'])
    train_dataset = train_data_loader.to_dataset()

    test_data_loader = SLSTRDataLoader(args, test_paths, batch_size=args['batch_size'], no_cache=args['no_cache'])
    test_dataset = test_data_loader.to_dataset()

    return train_dataset, test_dataset
