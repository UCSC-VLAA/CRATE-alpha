# Copyright 2022 Big Vision Authors.
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

"""TensorFlow Datasets as data source for big_vision."""
import functools

import datasets.core as ds_core
import jax
import overrides
import tensorflow_datasets as tfds

import tensorflow.io.gfile as gfile  # pylint: disable=consider-using-from-import
import io
import numpy as np
import tensorflow as tf

class DataSource(ds_core.DataSource):
  """Use TFDS as a data source."""

  def __init__(self, data_dir=None):

    self.data_dir = data_dir
    with gfile.GFile(data_dir, "rb") as f:
      data = f.read()
    self.loaded = np.load(io.BytesIO(data), allow_pickle=False)


    self.loaded_images = np.array_split(self.loaded['image'], jax.process_count())

    self.loaded_labels = np.array_split(self.loaded['label'], jax.process_count())

    # Each host is responsible for a fixed subset of data

    self.split_data = {'image':self.loaded_images[jax.process_index()],
                       'label': self.loaded_labels[jax.process_index()]
                       }


  @overrides.overrides
  def get_tfdata(self, ordered=False):
    train_dataset = tf.data.Dataset.from_tensor_slices(self.split_data)

    return train_dataset

  @property
  @overrides.overrides
  def total_examples(self):
    return self.loaded['image'].shape[0]

  @overrides.overrides
  def num_examples_per_process(self, nprocess=None):

    return   self.split_data['image'].shape[0]


