# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Code for SNIPS experiment.

Example command:
python snips_similarity_train.py \
 --data_dir=/my/path/snips_data/ACL2020data/preprocessed/5 \
 --few_shot=5 \
 --bert_config_file=/my/path/bert/bert_config.json \
 --vocab_file=/my/path/bert/vocab.txt \
 --init_checkpoint=/my/path/bert/bert_model.ckpt \
 --target_domain=AddToPlaylist \
 --use_tpu=false
"""

import csv
import json
import math
import os
import random
import sys
from absl import app
from absl import flags
from bert import modeling
from bert import optimization
from bert import tokenization
from language.common.utils import experiment_utils
from language.common.utils import tensor_utils
from language.common.utils import tpu_utils
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
DROPOUT_PROB = 0.2

# Required parameters
flags.DEFINE_string("data_dir", None,
                    "The directory with data preprocessing outputs.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 70,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("overwrite_data", True, "Overwrite stored tf record and read"
                  "from data_dir.")

flags.DEFINE_bool(
    "output_scalar", False, "Output to one scalar instead of "
    "projecting to the number of classes.")

flags.DEFINE_float("save_every_epoch", 1, "number of shots.")
# for few-shot
flags.DEFINE_integer("few_shot", 5, "number of shots.")
# continual learning setting
flags.DEFINE_string("continual_learning", "pretrain",
                    "setting for continual learning: pretrain, few_shot.")
flags.DEFINE_integer("known_num_shots", 100,
                     "number of shots for known classes.")

flags.DEFINE_integer(
    "n_way", 24,
    "number of classes for each episode: need to be devisible by 8 on tpus.")
flags.DEFINE_integer("n_support", 10,
                     "number of examples to form the support set.")
flags.DEFINE_integer("n_query", 10, "number of examples to form the query set.")
flags.DEFINE_string("emb_rep", "mean",
                    "embedding representation for bert encoding.")
flags.DEFINE_integer(
    "tpu_split", 8,
    "number of split in tpus, which is used for batch size of examples.")
flags.DEFINE_boolean(
    "few_shot_known_neg", False,
    "sample same number of few_shot from known examples to create negative"
    "examples.")
flags.DEFINE_boolean(
    "sample_dynamic", False,
    "if True, dynamically sample examples from the source domain instead of"
    "only choosing the first few examples from each class.")
flags.DEFINE_string(
    "reduce_method", "max",
    "reduce method to get one logit per label: max, mean, random")
flags.DEFINE_boolean(
    "min_max", False,
    "if True, reduce_method is min for the same class while max for different"
    "classes.")
flags.DEFINE_boolean("normalize", False,
                     "normalize embeddings before doing multiplication.")

# SNIPS parsing specific
flags.DEFINE_string("target_domain", None, "target domain.")

tapnet_domain_info = {
    "AddToPlaylist": {
        "src_max": 1491,
        "src_types": 32,
        "src_train_num": 47712
    },
    "BookRestaurant": {
        "src_max": 1395,
        "src_types": 31,
        "src_train_num": 43245
    },
    "GetWeather": {
        "src_max": 1804,
        "src_types": 30,
        "src_train_num": 54120
    },
    "PlayMusic": {
        "src_max": 1804,
        "src_types": 29,
        "src_train_num": 52316
    },
    "RateBook": {
        "src_max": 1395,
        "src_types": 30,
        "src_train_num": 41850
    },
    "SearchCreativeWork": {
        "src_max": 1395,
        "src_types": 35,
        "src_train_num": 48825
    },
    "SearchScreeningEvent": {
        "src_max": 1681,
        "src_types": 27,
        "src_train_num": 45387
    },
}

tapnet_domain_info_1_shot = {
    "AddToPlaylist": {
        "src_max": 1492,
        "src_types": 32,
        "src_train_num": 47744
    },
    "BookRestaurant": {
        "src_max": 1492,
        "src_types": 31,
        "src_train_num": 46252
    },
    "GetWeather": {
        "src_max": 2117,
        "src_types": 30,
        "src_train_num": 63510
    },
    "PlayMusic": {
        "src_max": 2117,
        "src_types": 29,
        "src_train_num": 61393
    },
    "RateBook": {
        "src_max": 1492,
        "src_types": 30,
        "src_train_num": 44760
    },
    "SearchCreativeWork": {
        "src_max": 1492,
        "src_types": 35,
        "src_train_num": 52220
    },
    "SearchScreeningEvent": {
        "src_max": 1986,
        "src_types": 27,
        "src_train_num": 53622
    },
}


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class SNIPSProcessor(DataProcessor):
  """Processor for the SNIPS data set."""

  def get_slot_list(self):
    """Gets a list of slot names."""
    slot_list = [
        "playlist", "music_item", "geographic_poi", "facility", "movie_name",
        "location_name", "restaurant_name", "track", "restaurant_type",
        "object_part_of_series_type", "country", "service", "poi",
        "party_size_description", "served_dish", "genre", "current_location",
        "object_select", "album", "object_name", "state", "sort",
        "object_location_type", "movie_type", "spatial_relation", "artist",
        "cuisine", "entity_name", "object_type", "playlist_owner", "timeRange",
        "city", "rating_value", "best_rating", "rating_unit", "year",
        "party_size_number", "condition_description", "condition_temperature"
    ]
    return slot_list

  def get_domain2slot(self):
    """Domain to slot mapping constants."""
    domain2slot = {
        "AddToPlaylist": [
            "music_item", "playlist_owner", "entity_name", "playlist", "artist"
        ],
        "BookRestaurant": [
            "city", "facility", "timeRange", "restaurant_name", "country",
            "cuisine", "restaurant_type", "served_dish", "party_size_number",
            "poi", "sort", "spatial_relation", "state", "party_size_description"
        ],
        "GetWeather": [
            "city", "state", "timeRange", "current_location", "country",
            "spatial_relation", "geographic_poi", "condition_temperature",
            "condition_description"
        ],
        "PlayMusic": [
            "genre", "music_item", "service", "year", "playlist", "album",
            "sort", "track", "artist"
        ],
        "RateBook": [
            "object_part_of_series_type", "object_select", "rating_value",
            "object_name", "object_type", "rating_unit", "best_rating"
        ],
        "SearchCreativeWork": ["object_name", "object_type"],
        "SearchScreeningEvent": [
            "timeRange", "movie_type", "object_location_type", "object_type",
            "location_name", "spatial_relation", "movie_name"
        ]
    }
    return domain2slot

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(data_dir)

  def get_labels(self):
    return self.get_slot_list()

  def _create_examples(self, input_data_path):
    """Creates examples for the training sets."""
    all_examples = {}
    with tf.io.gfile.GFile(input_data_path) as f:
      all_data = json.load(f)
      for data_type, data_value in all_data.items():
        examples = []
        tf.logging.info("building data for %s" % data_type)
        for idx, utterance_label in enumerate(data_value):
          guid = "%s-%s" % (data_type, idx)
          utterance, label = utterance_label
          text_a = tokenization.convert_to_unicode(utterance.strip())
          label = tokenization.convert_to_unicode(label.strip())

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        all_examples[data_type] = examples

    train_examples = all_examples["train"] + all_examples["oos_train"]
    random.shuffle(train_examples)

    return train_examples

  def get_dev_examples(self, data_dir):
    all_examples = {}
    with tf.io.gfile.GFile(data_dir) as f:
      all_data = json.load(f)
      for data_type, data_value in all_data.items():
        examples = []
        tf.logging.info("dev building data for %s" % data_type)
        for idx, utterance_label in enumerate(data_value):
          guid = "%s-%s" % (data_type, idx)
          utterance, label = utterance_label
          text_a = tokenization.convert_to_unicode(utterance.strip())
          label = tokenization.convert_to_unicode(label.strip())

          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        all_examples[data_type] = examples
    return all_examples["val"], all_examples["oos_val"]


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                num_train_classes, num_shots_per_class,
                                ft_known_train_file, ft_known_num_classes,
                                ft_known_num_shots, use_tpu):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "slot_name_id": tf.FixedLenFeature([], tf.int64),
      "start_i": tf.FixedLenFeature([], tf.int64),
      "end_i": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "real_seq_len": tf.FixedLenFeature([], tf.int64),
      "domain_id": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    tf.logging.info("input_function() params is training: %s" % is_training)
    tf.logging.info("input_function() params: %s", str(params))

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)

    if is_training:
      window_batch_size = num_shots_per_class
      num_classes = num_train_classes

      d = d.map(lambda record: _decode_record(record, name_to_features))

      # shuffle elements in each slot type
      # need to flat map first to deal with dict
      d = d.window(1)  # create dummy window to use zip
      d = d.flat_map(lambda d: tf.data.Dataset.zip({k: d[k] for k in d}))
      d = d.batch(window_batch_size, drop_remainder=True)

      def d_shuffle(features):
        shuffled_features = dict()
        indices = tf.range(start=0, limit=window_batch_size, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        for k in features.keys():
          shuffled_features[k] = tf.gather(features[k], shuffled_indices)
        return shuffled_features

      d = d.map(d_shuffle)
      d = d.unbatch()

      # group a1a2a3...akb1b2b3..bk...n1n2n3...nk to
      # a1b1...n1
      # a2b2...bk
      # ...
      # akbk...nk
      d = d.window(num_classes, 1, window_batch_size, True)
      d = d.flat_map(lambda w: tf.data.Dataset.zip(
          {k: w[k].batch(num_classes, drop_remainder=True) for k in w}))

      if FLAGS.few_shot_known_neg:
        known_d = tf.data.TFRecordDataset(ft_known_train_file)
        known_d = known_d.map(
            lambda record: _decode_record(record, name_to_features))

        sample_dynamic = FLAGS.sample_dynamic
        if sample_dynamic:
          known_d = known_d.window(1)
          known_d = known_d.flat_map(
              lambda m: tf.data.Dataset.zip({k: m[k] for k in m}))
          known_d = known_d.batch(ft_known_num_shots, drop_remainder=True)

          def known_d_shuffle(features):
            shuffled_features = dict()
            indices = tf.range(
                start=0, limit=ft_known_num_shots, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            for k in features.keys():
              shuffled_features[k] = tf.gather(features[k], shuffled_indices)
            return shuffled_features

          known_d = known_d.map(known_d_shuffle)
          known_d = known_d.unbatch()

        # Get the first few_shot elements from known data and shuffle elements
        # in each slot type
        known_d = known_d.window(num_shots_per_class, ft_known_num_shots, True)
        known_d = known_d.flat_map(
            lambda m: tf.data.Dataset.zip({k: m[k] for k in m}))
        known_d = known_d.batch(num_shots_per_class, drop_remainder=True)
        known_d = known_d.map(d_shuffle)
        known_d = known_d.unbatch()
        known_d = known_d.window(ft_known_num_classes, 1, num_shots_per_class,
                                 True)
        known_d = known_d.flat_map(lambda w: tf.data.Dataset.zip({
            k: w[k].batch(ft_known_num_shots, drop_remainder=True) for k in w
        }))

        zip_data = tf.data.Dataset.zip((d, known_d))

        def combine(features):
          unknown_features = features[0]
          known_features = features[1]
          combined_features = dict()
          for k in unknown_features.keys():
            combined_features[k] = tf.concat(
                (unknown_features[k], known_features[k]), axis=0)
          return combined_features

        d = zip_data.map(lambda x, y: combine((x, y)))

      # For tpus, we need batch size to be divisible by 8
      # pad only if num_classes < 8 so that we can ignore padding
      tpu_split = FLAGS.tpu_split if use_tpu else 1
      if window_batch_size < tpu_split:
        required_batch_num = math.ceil(
            window_batch_size / tpu_split) * tpu_split
        padded_batch_num = required_batch_num - window_batch_size
        padding_batch = d.take(1).repeat(padded_batch_num)

        def set_padding_batch_real(features):
          new_features = dict()
          for k in features.keys():
            if k == "is_real_example":
              new_features[k] = features[k] * 0
            else:
              new_features[k] = features[k]
          return new_features

        padding_batch = padding_batch.map(set_padding_batch_real)
        d = d.concatenate(padding_batch)

      d = d.repeat()

      if use_tpu:
        d = d.batch(tpu_split, drop_remainder=True)
        # now each batch has shape: 8 x num_classes

    else:
      assert False, "Not Implemented"

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, slot_names, start_i, end_i, use_one_hot_embeddings,
                 num_classes, num_shots_per_class, ft_known_num_classes,
                 use_tpu):
  """Creates a classification model."""
  tpu_split = FLAGS.tpu_split if use_tpu else 1
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_final_layer = model.get_sequence_output()

  # shape: bze, max_seq_len, hidden
  if FLAGS.emb_rep == "cls":
    embedding = tf.squeeze(output_final_layer[:, 0:1, :], axis=1)
  elif FLAGS.emb_rep == "mean":
    embedding = tf.reduce_mean(output_final_layer, axis=1)

  print_op = tf.print(
      "start shape",
      start_i.shape,
      "\n emb shape",
      embedding.shape,
      # "\n emb", embedding[:3, :10],
      output_stream=tf.logging.info,
      summarize=-1)
  with tf.control_dependencies([print_op]):
    embedding = embedding * 1

  # output_final_layer: bze, seq_len, hid_size
  # start_i:bze
  emb_bze = output_final_layer.shape[0].value
  indices_0 = tf.expand_dims(tf.range(emb_bze), axis=1)
  start_indices = tf.expand_dims(start_i, axis=1)
  end_indices = tf.expand_dims(end_i, axis=1)
  emb_start_indices = tf.concat([indices_0, start_indices], axis=1)
  emb_end_indices = tf.concat([indices_0, end_indices], axis=1)
  start_embedding = tf.gather_nd(output_final_layer,
                                 emb_start_indices)  # bze, hid_size
  end_embedding = tf.gather_nd(output_final_layer, emb_end_indices)

  tf.logging.info("per tpu slice")
  tf.logging.info("emebdding size: %s", embedding.shape)
  tf.logging.info("label size: %s", labels.shape)

  tf.logging.info("start emebdding size: %s", start_embedding.shape)
  tf.logging.info("end emebdding size: %s", end_embedding.shape)
  tf.logging.info("start_i size: %s", start_i.shape)
  tf.logging.info("=======" * 10)

  # for tpu usage: combine embeddings after splitting 8 ways
  # [global_batch_size]
  if use_tpu:
    labels = tpu_utils.cross_shard_concat(labels)
    tf.logging.info("label size: %s", labels.shape)
    tf.logging.info("=======" * 10)
    # [global_batch_size, hidden_size]
    embedding = tpu_utils.cross_shard_concat(embedding)
    start_embedding = tpu_utils.cross_shard_concat(start_embedding)
    end_embedding = tpu_utils.cross_shard_concat(end_embedding)
    slot_names = tpu_utils.cross_shard_concat(slot_names)
  tf.logging.info("Global batch size: %s", tensor_utils.shape(embedding, 0))

  tf.logging.info("emebdding size: %s", embedding.shape)
  tf.logging.info("label size: %s", labels.shape)
  tf.logging.info("start emebdding size: %s", start_embedding.shape)
  tf.logging.info("end emebdding size: %s", end_embedding.shape)
  tf.logging.info("slot_name size: %s", slot_names.shape)
  tf.logging.info("num tpu shards: %s", tpu_utils.num_tpu_shards())
  tf.logging.info("=======" * 10)

  n_examples = num_shots_per_class
  if FLAGS.few_shot_known_neg:
    real_num_classes = num_classes
    num_classes += ft_known_num_classes

  # Note:
  # I. if training on TPUs, input is distributed into 8 slices, and each slices
  # is padded independently to batch_size (128 by default). So we need to 1.
  # remove paddings in each slice, and 2. remove padded batches (to be 8 if
  # few_shot < 8).
  # II. if training without distributed, the whole input is padded to batzh_size
  # so we only need to remove the padded part

  if use_tpu:
    real_shift = math.ceil(num_classes / FLAGS.batch_size) * FLAGS.batch_size
    # if use TPU, then embedding.shape[0] will be (num_classes + pad_num) * 8
    real_indices = tf.range(num_classes)
    for i in range(1, tpu_split):
      real_indices = tf.concat(
          [real_indices, tf.range(num_classes) + real_shift * i], axis=0)
    embedding = tf.gather(embedding, real_indices)
    labels = tf.gather(labels, real_indices)
    tf.logging.info("emebdding size after removing padding in batch: %s",
                    embedding.shape)
    tf.logging.info("label size after removing padding in batch: %s",
                    labels.shape)
    start_embedding = tf.gather(start_embedding, real_indices)
    end_embedding = tf.gather(end_embedding, real_indices)
    slot_names = tf.gather(slot_names, real_indices)
    tf.logging.info("start emebdding size after removing padding in batch: %s",
                    start_embedding.shape)
    tf.logging.info("end emebdding size after removing padding in batch: %s",
                    end_embedding.shape)
    tf.logging.info("slot names size after removing padding in batch: %s",
                    slot_names.shape)
    # remove tpu padding
    if n_examples < tpu_split:
      real_batch_total = n_examples * num_classes
      embedding = embedding[:real_batch_total]
      labels = labels[:real_batch_total]
      start_embedding = start_embedding[:real_batch_total]
      end_embedding = end_embedding[:real_batch_total]
      slot_names = slot_names[:real_batch_total]
      real_num = n_examples
    else:
      real_num = tpu_split
  else:
    # not use TPUs
    if n_examples < tpu_split:
      real_num = n_examples
    else:
      real_num = tpu_split
    real_batch_total = real_num * num_classes

    embedding = embedding[:real_batch_total]
    labels = labels[:real_batch_total]

    start_embedding = start_embedding[:real_batch_total]
    end_embedding = end_embedding[:real_batch_total]
    slot_names = slot_names[:real_batch_total]

  tf.logging.info("real embedding size: %s", embedding.shape)
  tf.logging.info("real label size: %s", labels.shape)

  tf.logging.info("real start emebdding size: %s", start_embedding.shape)
  tf.logging.info("real end emebdding size: %s", end_embedding.shape)
  tf.logging.info("real slot names size: %s", slot_names.shape)

  n = embedding.shape[0].value
  assert n == real_num * num_classes, "n: %d; real_num: %d: num_classes: %d" % (
      n, real_num, num_classes)

  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    if is_training:
      # I.e., 0.1 dropout
      embedding = tf.nn.dropout(embedding, keep_prob=1 - DROPOUT_PROB)
      start_embedding = tf.nn.dropout(
          start_embedding, keep_prob=1 - DROPOUT_PROB)
      end_embedding = tf.nn.dropout(end_embedding, keep_prob=1 - DROPOUT_PROB)

    embedding = tf.concat((start_embedding, end_embedding),
                          axis=1)  # bze, 2 x hid_size
    tf.logging.info("++++" * 20)
    tf.logging.info("endpoint embedding size: %s", embedding.shape)

    if FLAGS.normalize:
      embedding = tf.math.l2_normalize(embedding, axis=-1)

    logits = tf.matmul(embedding, embedding, transpose_b=True)
    diagonal_matrix = tf.eye(n, n)
    logits = logits - diagonal_matrix * logits

    logits_reshape = tf.reshape(logits, [n, real_num, num_classes])

    if FLAGS.reduce_method == "mean":
      all_logits_sum = tf.reduce_sum(logits_reshape, 1)
      num_counts = tf.ones([n, num_classes]) * real_num
      label_diagonal = tf.eye(num_classes, num_classes)
      label_diagonal = tf.tile(label_diagonal, tf.constant([real_num, 1]))
      num_counts = num_counts - label_diagonal
      mean_logits = tf.divide(all_logits_sum, num_counts)
      if FLAGS.few_shot_known_neg:
        real_logits_indices = tf.range(real_num_classes)
        for i in range(1, real_num):
          real_logits_indices = tf.concat([
              real_logits_indices,
              tf.range(real_num_classes) + num_classes * i
          ],
                                          axis=0)
        mean_logits = tf.gather(mean_logits, real_logits_indices)

        label_diagonal = tf.eye(real_num_classes, num_classes)
        label_diagonal = tf.tile(label_diagonal, tf.constant([real_num, 1]))

      probabilities = tf.nn.softmax(mean_logits, axis=-1)
      log_probs = tf.nn.log_softmax(mean_logits, axis=-1)
      return_logits = mean_logits

    elif FLAGS.reduce_method == "max":
      max_logits = tf.reduce_max(logits_reshape, 1)

      if FLAGS.min_max:
        # because the diagnoal is 0, we need to assign a large number to get the
        # true min
        large_num = 50000
        added_logits = logits + diagonal_matrix * large_num
        added_reshape_logits = tf.reshape(added_logits,
                                          [n, real_num, num_classes])
        min_logits = tf.reduce_min(added_reshape_logits, 1)  # n * num_classes
        masks = tf.tile(
            tf.eye(num_classes, num_classes), tf.constant([real_num, 1]))
        max_logits = masks * min_logits + (1 - masks) * max_logits

      label_diagonal = tf.eye(num_classes, num_classes)

      if FLAGS.few_shot_known_neg:
        real_logits_indices = tf.range(real_num_classes)
        for i in range(1, real_num):
          real_logits_indices = tf.concat([
              real_logits_indices,
              tf.range(real_num_classes) + num_classes * i
          ],
                                          axis=0)
        max_logits = tf.gather(max_logits, real_logits_indices)
        label_diagonal = label_diagonal[:real_num_classes]

      label_diagonal = tf.tile(label_diagonal, tf.constant([real_num, 1]))

      probabilities = tf.nn.softmax(max_logits, axis=-1)
      log_probs = tf.nn.log_softmax(max_logits, axis=-1)
      return_logits = max_logits

    elif FLAGS.reduce_method == "random":
      indice_0 = tf.expand_dims(tf.range(n), axis=1)  # n x 1
      indice_1 = tf.random.uniform([n, 1],
                                   minval=0,
                                   maxval=real_num,
                                   dtype=tf.dtypes.int32)
      random_indices = tf.concat([indice_0, indice_1], axis=1)
      random_logits = tf.gather_nd(logits_reshape, random_indices)

      label_diagonal = tf.eye(num_classes, num_classes)

      if FLAGS.few_shot_known_neg:
        real_logits_indices = tf.range(real_num_classes)
        for i in range(1, real_num):
          real_logits_indices = tf.concat([
              real_logits_indices,
              tf.range(real_num_classes) + num_classes * i
          ],
                                          axis=0)
        random_logits = tf.gather(random_logits, real_logits_indices)
        label_diagonal = label_diagonal[:real_num_classes]

      label_diagonal = tf.tile(label_diagonal, tf.constant([real_num, 1]))

      probabilities = tf.nn.softmax(random_logits, axis=-1)
      log_probs = tf.nn.log_softmax(random_logits, axis=-1)
      return_logits = random_logits
    per_example_loss = -tf.reduce_sum(label_diagonal * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, return_logits, probabilities)


def pad_feature(features):
  """Adds padding to tensors in features."""
  # """WARNING: batch size can be too much for memory by doing this way"""
  new_input_features = dict()
  input_features = ["input_ids", "input_mask", "segment_ids",
                    "label_ids"]  # three-dim

  # NOTE: on tpus, n_way is split by 8. So n_way here is not the same as
  # FLAGS.n_way need to pad to batch_size (divisible by 32) for TPUs.
  for k in features.keys():
    if k in input_features:
      # NOTE: tensor.shape returns Dimensions instead of int.
      bze, n_classes, input_dim = features[k].get_shape().as_list()
      reshape_batch_size = bze * n_classes
      reshaped_feature = tf.reshape(features[k],
                                    [reshape_batch_size, input_dim])
      num_pad = math.ceil(reshape_batch_size / FLAGS.batch_size
                         ) * FLAGS.batch_size - reshape_batch_size
      pad_zeros = tf.zeros([num_pad, input_dim], dtype=tf.dtypes.int32)
      new_input_features[k] = tf.concat([reshaped_feature, pad_zeros], axis=0)
    else:
      bze, n_classes = features[k].get_shape().as_list()
      reshape_batch_size = bze * n_classes
      reshaped_feature = tf.reshape(features[k], [reshape_batch_size])
      num_pad = math.ceil(reshape_batch_size / FLAGS.batch_size
                         ) * FLAGS.batch_size - reshape_batch_size
      pad_zeros = tf.zeros([num_pad], dtype=tf.dtypes.int32)
      new_input_features[k] = tf.concat([reshaped_feature, pad_zeros], axis=0)
  return new_input_features


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, num_classes, num_shots_per_class,
                     ft_known_num_classes):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    del params  # unused
    del labels  # unused

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    tf.logging.info("*** Features before padding ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    if is_training:
      # for tpu
      if use_tpu:
        features = pad_feature(features)

      tf.logging.info("*** Features after padding ***")
      for name in sorted(features):
        tf.logging.info("  name = %s, shape = %s" %
                        (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    slot_name = features["slot_name_id"]
    start_i = features["start_i"]
    end_i = features["end_i"]
    is_real_example = None

    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    (total_loss, per_example_loss, logits,
     _) = create_model(bert_config, is_training, input_ids, input_mask,
                       segment_ids, label_ids, slot_name, start_i, end_i,
                       use_one_hot_embeddings, num_classes, num_shots_per_class,
                       ft_known_num_classes, use_tpu)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map,
       _) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    print_op = tf.print(
        "tf_total_loss", total_loss, output_stream=sys.stdout, summarize=-1)
    with tf.control_dependencies([print_op]):
      total_loss = total_loss * 1
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          loss=total_loss,
          init_lr=learning_rate,
          num_train_steps=num_train_steps,
          num_warmup_steps=num_warmup_steps,
          use_tpu=use_tpu)
      if use_tpu:
        output_spec = tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            train_op=train_op,
            scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        if FLAGS.output_scalar:
          predictions = tf.cast(
              tf.sigmoid(logits) > FLAGS.sim_threshold, dtype=tf.int32)
        else:
          predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

        accuracy = tf.metrics.accuracy(
            labels=label_ids[:, 0],
            predictions=predictions,
            weights=is_real_example)

        if FLAGS.output_scalar:
          return {
              "eval_accuracy": accuracy,
          }

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = metric_fn(per_example_loss, label_ids, logits,
                               is_real_example)
      if use_tpu:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=eval_metrics,
        )

    return output_spec

  return model_fn


def _get_assets_extra():
  """Gets a dict that maps from filenames to paths to be exported."""
  assets_extra = {
      "bert_config.json": FLAGS.bert_config_path,
      "wordpiece.vocab": FLAGS.vocab_file,
  }
  return assets_extra


def _get_hparams():
  """Create dictionary of hperparameters."""
  # NB: params will be used in `model_fn`, `input_fn`, and
  # `serving_input_receiver_fn`.
  params = {
      "dropout_prob": DROPOUT_PROB,
      "learning_rate": FLAGS.learning_rate,
      "num_train_steps": FLAGS.num_train_steps,
      "use_tpu": FLAGS.use_tpu,
      "batch_size": FLAGS.batch_size,
      "eval_batch_size": FLAGS.eval_batch_size,
  }
  return params


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.config.set_soft_device_placement(True)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  processor = SNIPSProcessor()
  label_list = processor.get_labels()  # note: slot_name only, no B/I/O

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  domain_info = tapnet_domain_info
  if FLAGS.few_shot == 1:
    domain_info = tapnet_domain_info_1_shot

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  train_input_fn = None
  ft_known_train_file = None
  ft_known_num_classes = None
  ft_known_num_shots = None
  if FLAGS.do_train:
    if FLAGS.continual_learning is None:
      assert False, "Not Implemented"
    elif FLAGS.continual_learning == "pretrain":
      train_file = os.path.join(FLAGS.data_dir, FLAGS.target_domain,
                                "train.tf_record")
      num_classes = domain_info[FLAGS.target_domain]["src_types"]
      num_train_examples = domain_info[FLAGS.target_domain]["src_train_num"]
      num_shots_per_class = domain_info[FLAGS.target_domain]["src_max"]
    elif FLAGS.continual_learning == "few_shot":
      train_file = os.path.join(FLAGS.data_dir, FLAGS.target_domain,
                                "%s_train.tf_record" % FLAGS.target_domain)
      num_classes = domain_info[FLAGS.target_domain]["tgt_types"]
      num_train_examples = domain_info[FLAGS.target_domain]["tgt_train_num"]
      num_shots_per_class = domain_info[FLAGS.target_domain]["tgt_max"]
      # for negative samples with src domain
      ft_known_train_file = os.path.join(FLAGS.data_dir, FLAGS.target_domain,
                                         "src_support.tf_record")
      ft_known_num_classes = domain_info[FLAGS.target_domain]["src_types"]
      ft_known_num_shots = domain_info[FLAGS.target_domain]["src_max"]
    tpu_split = FLAGS.tpu_split if FLAGS.use_tpu else 1
    if num_shots_per_class < tpu_split:
      steps_per_epoch = 1
    else:
      steps_per_epoch = num_shots_per_class // tpu_split
    num_train_steps = int(steps_per_epoch * FLAGS.num_train_epochs)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    FLAGS.num_train_steps = num_train_steps

    FLAGS.save_checkpoints_steps = int(steps_per_epoch * FLAGS.save_every_epoch)

    if not tf.gfile.Glob(train_file):
      assert False, "train_file: %s not exists" % train_file

    tf.logging.info("***** Running training *****")
    tf.logging.info("  train_file: %s" % train_file)
    tf.logging.info("  use_tpu: %s" % FLAGS.use_tpu)
    tf.logging.info("  Num examples = %d", num_train_examples)
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)
    tf.logging.info("  Save checkpoints steps = %d",
                    FLAGS.save_checkpoints_steps)
    tf.logging.info("  warmup steps = %d", num_warmup_steps)
    tf.logging.info("  Num epochs = %d", FLAGS.num_train_epochs)
    tf.logging.info("  Num steps = %d", num_train_steps)
    tf.logging.info("  Reduce method = %s", FLAGS.reduce_method)
    tf.logging.info("  Max Seq Length = %d", FLAGS.max_seq_length)
    tf.logging.info(" learning_rate = %.7f", FLAGS.learning_rate)
    tf.logging.info(" dropout rate = %.4f", DROPOUT_PROB)

    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        num_train_classes=num_classes,
        num_shots_per_class=num_shots_per_class,
        ft_known_train_file=ft_known_train_file,
        ft_known_num_classes=ft_known_num_classes,
        ft_known_num_shots=ft_known_num_shots,
        use_tpu=FLAGS.use_tpu,
    )

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      num_classes=num_classes,
      num_shots_per_class=num_shots_per_class,
      ft_known_num_classes=ft_known_num_classes,
  )

  FLAGS.do_eval = False
  params = _get_hparams()
  params.update(num_train_steps=num_train_steps)

  experiment_utils.run_experiment(
      model_fn=model_fn,
      train_input_fn=train_input_fn,
      eval_input_fn=train_input_fn,
      params=params)


if __name__ == "__main__":
  app.run(main)
