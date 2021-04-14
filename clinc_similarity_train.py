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
r"""Code for CLINC experiment.

Example command:
python clinc_similarity_train.py \
 --data_dir=/my/path/oos-eval/data \
 --data_output_dir=/my/path/oos-eval/preprocessed \
 --bert_config_file=/my/path/bert/bert_config.json \
 --vocab_file=/my/path/bert/vocab.txt \
 --init_checkpoint=/my/path/bert/bert_model.ckpt \
 --use_tpu=false
"""

import collections
import json
import math
import os
import random
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
NUM_CLASSES = 151
DROPOUT_PROB = 0.2

# Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the data_full.json file.")

flags.DEFINE_string(
    "data_output_dir", None,
    "The directory with data preprocessing outputs, e.g. data split. If the"
    "directory is empty, the preprocessing step will be run.")

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
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

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

flags.DEFINE_bool(
    "output_scalar", False, "Output to one scalar instead of "
    "projecting to the number of classes")
flags.DEFINE_float("sim_threshold", 0.5,
                   "threshold for similarity after sigmoid for eval")

flags.DEFINE_integer("save_every_epoch", 1, "number of shots")
# for few-shot
flags.DEFINE_integer("few_shot", 5, "number of shots")
# continual learning setting
flags.DEFINE_string("continual_learning", "pretrain",
                    "setting for continual learning: pretrain, few_shot")
flags.DEFINE_integer("num_domains", 10, "number of known domains for pretrain")
flags.DEFINE_integer("num_labels_per_domain", 5,
                     "number of intents from each known domain for pretrain")
flags.DEFINE_integer("known_num_shots", 100,
                     "number of shots for known classes")

flags.DEFINE_integer(
    "n_way", 24,
    "number of classes for each episode: need to be devisible by 8 on tpus")
flags.DEFINE_integer("n_support", 10,
                     "number of examples to form the support set")
flags.DEFINE_integer("n_query", 10, "number of examples to form the query set")
flags.DEFINE_string("emb_rep", "mean",
                    "embedding representation for bert encoding")
flags.DEFINE_integer(
    "tpu_split", 8,
    "number of split in tpus, which is used for batch size of examples")
flags.DEFINE_boolean(
    "few_shot_known_neg", False,
    "sample same number of few_shot from known examples to create negative examples"
)
flags.DEFINE_boolean(
    "sample_dynamic", False,
    "if True, dynamically sample examples from the source domain instead of"
    "only choosing the first few examples from each class")
flags.DEFINE_string(
    "reduce_method", "max",
    "reduce method to get one logit per label: max, mean, random")
flags.DEFINE_boolean(
    "min_max", False,
    "if True, reduce_method is min for the same class while max for different classes"
)


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


def convert_single_example(ex_index, example, label_map, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" %
                    " ".join([tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def file_based_convert_examples_to_features(examples, label_map,
                                            max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                ft_known_train_file, use_tpu):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
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
      num_known_classes = FLAGS.num_domains * FLAGS.num_labels_per_domain
      num_unknown_classes = NUM_CLASSES - num_known_classes
      if FLAGS.continual_learning == "pretrain":
        window_batch_size = FLAGS.known_num_shots
        num_classes = num_known_classes
      elif FLAGS.continual_learning == "few_shot":
        window_batch_size = FLAGS.few_shot
        num_classes = num_unknown_classes

      d = d.map(lambda record: _decode_record(record, name_to_features))

      d = d.take(num_classes * window_batch_size)  # remove padded examples

      # shuffle elements in each intent
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
        known_d = known_d.take(num_known_classes *
                               FLAGS.known_num_shots)  # remove padded examples

        sample_dynamic = FLAGS.sample_dynamic
        if sample_dynamic:
          known_d = known_d.window(1)
          known_d = known_d.flat_map(
              lambda m: tf.data.Dataset.zip({k: m[k] for k in m}))
          known_d = known_d.batch(FLAGS.known_num_shots, drop_remainder=True)

          def known_d_shuffle(features):
            shuffled_features = dict()
            indices = tf.range(
                start=0, limit=FLAGS.known_num_shots, dtype=tf.int32)
            shuffled_indices = tf.random.shuffle(indices)
            for k in features.keys():
              shuffled_features[k] = tf.gather(features[k], shuffled_indices)
            return shuffled_features

          known_d = known_d.map(known_d_shuffle)
          known_d = known_d.unbatch()

        # Get the first few_shot elements from known data and shuffle elements
        # in each intent
        known_d = known_d.window(FLAGS.few_shot, FLAGS.known_num_shots, True)
        known_d = known_d.flat_map(
            lambda m: tf.data.Dataset.zip({k: m[k] for k in m}))
        known_d = known_d.batch(FLAGS.few_shot, drop_remainder=True)
        known_d = known_d.map(d_shuffle)
        known_d = known_d.unbatch()
        known_d = known_d.window(num_known_classes, 1, FLAGS.few_shot, True)
        known_d = known_d.flat_map(lambda w: tf.data.Dataset.zip(
            {k: w[k].batch(num_known_classes, drop_remainder=True) for k in w}))

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

      # for tpus, we need batch size to be divisible by 8
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
                 labels, use_one_hot_embeddings, use_tpu):
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

  tf.logging.info("per tpu slice")
  tf.logging.info("emebdding size: %s", embedding.shape)
  tf.logging.info("label size: %s", labels.shape)
  tf.logging.info("=======" * 10)

  if use_tpu:
    # for tpu usage: combine embeddings after splitting 8 ways
    # [global_batch_size]
    labels = tpu_utils.cross_shard_concat(labels)
    tf.logging.info("label size: %s", labels.shape)
    tf.logging.info("=======" * 10)

    # [global_batch_size, hidden_size]
    embedding = tpu_utils.cross_shard_concat(embedding)

  tf.logging.info("Global batch size: %s", tensor_utils.shape(embedding, 0))

  tf.logging.info("emebdding size: %s", embedding.shape)
  tf.logging.info("label size: %s", labels.shape)
  tf.logging.info("num tpu shards: %s", tpu_utils.num_tpu_shards())
  tf.logging.info("=======" * 10)

  num_known_classes = FLAGS.num_domains * FLAGS.num_labels_per_domain
  num_unknown_classes = NUM_CLASSES - num_known_classes
  if FLAGS.continual_learning == "pretrain":
    num_classes = num_known_classes
    n_examples = FLAGS.known_num_shots
  elif FLAGS.continual_learning == "few_shot":
    num_classes = num_unknown_classes
    n_examples = FLAGS.few_shot

  if FLAGS.few_shot_known_neg:
    num_classes = NUM_CLASSES
    real_num_classes = num_unknown_classes

  # remove padding in each batch
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

    # remove padded batch
    if n_examples < tpu_split:
      real_batch_total = n_examples * num_classes
      embedding = embedding[:real_batch_total]
      labels = labels[:real_batch_total]
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

  tf.logging.info("real emebdding size: %s", embedding.shape)
  tf.logging.info("real label size: %s", labels.shape)

  n = embedding.shape[0].value

  assert n == real_num * num_classes, "n: %d; real_num: %d: num_classes: %d" % (
      n, real_num, num_classes)

  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
    if is_training:
      # I.e., 0.1 dropout
      embedding = tf.nn.dropout(embedding, keep_prob=1 - DROPOUT_PROB)

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
        for i in range(1, n_examples):
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
        # Because the diagnoal is 0, we need to assign a large number to get the
        # true min.
        large_number = 50000
        added_logits = logits + diagonal_matrix * large_number
        added_reshape_logits = tf.reshape(added_logits,
                                          [n, real_num, num_classes])
        min_logits = tf.reduce_min(added_reshape_logits, 1)  # n * num_classes
        masks = tf.tile(
            tf.eye(num_classes, num_classes), tf.constant([real_num, 1]))
        max_logits = masks * min_logits + (1 - masks) * max_logits

      label_diagonal = tf.eye(num_classes, num_classes)

      if FLAGS.few_shot_known_neg:
        real_logits_indices = tf.range(real_num_classes)
        # WARNING: current implementation may not be correct for few_shot > 8 on
        # tpus in the following for loop, it should be for i in
        # range(1, real_num) instead of in range(1, n_examples).
        assert n_examples < 8, ("current implementation may not be correct for "
                                "few_shot > 8 on tpus. Need to check")
        # Note: n_examples here is 2 or 5, which is less than tpu_slit.
        for i in range(1, n_examples):
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
        for i in range(1, n_examples):
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
  """WARNING: batch size can be too much for memory by doing this way."""
  new_input_features = dict()
  input_features = ["input_ids", "input_mask", "segment_ids"]  # three-dim

  # NOTE: on tpus, n_way is split by 8. So n_way here is not the same as
  # FLAGS.n_way need to pad to batch_size (divisible by 32) for TPUs.
  for k in features.keys():
    if k in input_features:
      # NOTE: tensor.shape returns Dimensions instead of int
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
                     use_one_hot_embeddings):
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
      if use_tpu:
        # for tpu
        features = pad_feature(features)
      tf.logging.info("*** Features after padding ***")
      for name in sorted(features):
        tf.logging.info("  name = %s, shape = %s" %
                        (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None

    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    (total_loss, per_example_loss, logits,
     _) = create_model(bert_config, is_training, input_ids, input_mask,
                       segment_ids, label_ids, use_one_hot_embeddings, use_tpu)

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
            labels=label_ids, predictions=predictions, weights=is_real_example)

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


def get_data(input_data_path):
  """Get examples and label_list from OOS input data."""
  all_examples = {}
  label_list = []
  with tf.io.gfile.GFile(input_data_path) as f:
    all_data = json.load(f)
    # train, val, test, oos_train, oos_val, oos_test
    for data_type, data_value in all_data.items():
      examples = []
      tf.logging.info("building data for %s" % data_type)
      for idx, utterance_label in enumerate(data_value):
        guid = "%s-%s" % (data_type, idx)
        utterance, label = utterance_label

        if label not in label_list:
          label_list.append(label)

        examples.append(
            InputExample(guid=guid, text_a=utterance, text_b=None, label=label))

      all_examples[data_type] = examples

  tf.logging.info("finished buidling data. Got %d labels" % len(label_list))
  return all_examples, label_list


def get_domain_map():
  """Constants of domain mapping."""
  domain_intent_map = {
      "meta": [
          "whisper_mode", "maybe", "change_language", "user_name",
          "change_user_name", "repeat", "sync_device", "cancel", "yes",
          "reset_settings", "change_accent", "change_speed", "no",
          "change_volume", "change_ai_name"
      ],
      "home": [
          "order_status", "order", "play_music", "todo_list", "reminder",
          "calendar", "smart_home", "todo_list_update", "reminder_update",
          "next_song", "shopping_list_update", "update_playlist",
          "calendar_update", "shopping_list", "what_song"
      ],
      "small_talk": [
          "where_are_you_from", "who_do_you_work_for", "what_is_your_name",
          "greeting", "do_you_have_pets", "are_you_a_bot", "who_made_you",
          "tell_joke", "meaning_of_life", "what_can_i_ask_you",
          "what_are_your_hobbies", "thank_you", "how_old_are_you", "goodbye",
          "fun_fact"
      ],
      "travel": [
          "plug_type", "car_rental", "flight_status", "book_flight",
          "international_visa", "vaccines", "lost_luggage",
          "travel_notification", "travel_suggestion", "carry_on", "timezone",
          "exchange_rate", "translate", "book_hotel", "travel_alert"
      ],
      "credit_card": [
          "expiration_date", "apr", "redeem_rewards", "credit_limit_change",
          "rewards_balance", "card_declined", "credit_limit",
          "application_status", "new_card", "report_lost_card", "damaged_card",
          "improve_credit_score", "international_fees",
          "replacement_card_duration", "credit_score"
      ],
      "work": [
          "direct_deposit", "pto_request_status", "insurance",
          "schedule_meeting", "rollover_401k", "next_holiday", "taxes",
          "pto_used", "meeting_schedule", "pto_request", "insurance_change",
          "pto_balance", "income", "w2", "payday"
      ],
      "utility": [
          "calculator", "timer", "measurement_conversion", "share_location",
          "find_phone", "spelling", "time", "date", "alarm", "roll_dice",
          "make_call", "definition", "text", "weather", "flip_coin"
      ],
      "auto": [
          "uber", "last_maintenance", "mpg", "tire_pressure", "traffic",
          "oil_change_how", "tire_change", "jump_start", "current_location",
          "gas_type", "distance", "directions", "oil_change_when", "gas",
          "schedule_maintenance"
      ],
      "banking": [
          "bill_balance", "bill_due", "transfer", "balance", "freeze_account",
          "account_blocked", "min_payment", "routing", "order_checks",
          "transactions", "interest_rate", "report_fraud", "pin_change",
          "pay_bill", "spending_history"
      ],
      "dining": [
          "food_last", "ingredients_list", "recipe", "restaurant_suggestion",
          "meal_suggestion", "restaurant_reviews", "cancel_reservation",
          "accept_reservations", "ingredient_substitution", "cook_time",
          "nutrition_info", "restaurant_reservation", "how_busy", "calories",
          "confirm_reservation"
      ],
  }
  return domain_intent_map


def save_ft_data(known_train_examples_by_label, unknown_train_examples_by_label,
                 num_few_shot, known_ft_output, unknown_ft_output, label_map,
                 max_seq_len, tokenizer, known_num_shots, batch_size):
  """Saves data for finetuning."""
  known_train_examples = []
  unknown_train_examples = []
  for _, labled_examples in known_train_examples_by_label.items():
    known_train_examples += labled_examples[:known_num_shots]
  for _, labled_examples in unknown_train_examples_by_label.items():
    unknown_train_examples += labled_examples[:num_few_shot]

  # padding for training on TPUs
  while len(known_train_examples) % batch_size != 0:
    known_train_examples.append(PaddingInputExample())
  while len(unknown_train_examples) % batch_size != 0:
    unknown_train_examples.append(PaddingInputExample())

  file_based_convert_examples_to_features(known_train_examples, label_map,
                                          max_seq_len, tokenizer,
                                          known_ft_output)
  file_based_convert_examples_to_features(unknown_train_examples, label_map,
                                          max_seq_len, tokenizer,
                                          unknown_ft_output)


def preprocess_few_shot_training_data(tokenizer, known_ft_path, unknown_ft_path,
                                      current_seed):
  """Preprocesses training data for few-shot experiments."""
  random.seed(current_seed)

  max_seq_len = FLAGS.max_seq_length
  all_examples, label_list = get_data(
      os.path.join(FLAGS.data_dir, "data_full.json"))
  # NOTE: label_list contains "OOS" label, which is not in the domain_intent_map
  domain_intent_map = get_domain_map()
  known_classes = []
  # There are 10 domains, 15 intents in each domain

  domain_list = list(domain_intent_map.keys())
  random.shuffle(domain_list)
  for domain in domain_list[:FLAGS.num_domains]:
    current_domain_labels = domain_intent_map[domain]
    random.shuffle(current_domain_labels)
    known_classes = (
        known_classes + current_domain_labels[:FLAGS.num_labels_per_domain])

  known_examples = collections.defaultdict(list)
  unknown_examples = collections.defaultdict(list)
  for example in all_examples["train"]:
    if example.label in known_classes:
      known_examples[example.label].append(example)
    else:
      unknown_examples[example.label].append(example)

  for oos_example in all_examples["oos_train"]:
    unknown_examples[oos_example.label].append(oos_example)

  shuffled_known_train_examples = dict()
  shuffled_unknown_train_examples = dict()
  for label, label_examples in known_examples.items():
    random.shuffle(label_examples)
    shuffled_known_train_examples[label] = label_examples
  for label, label_examples in unknown_examples.items():
    random.shuffle(label_examples)
    shuffled_unknown_train_examples[label] = label_examples

  tf.logging.info("num_known_classes: %s" % len(shuffled_known_train_examples))
  tf.logging.info("num_unknown_classes: %s" %
                  len(shuffled_unknown_train_examples))
  tf.logging.info("known_classes: %s" % known_classes[:5])
  # In order for val, test to be consistent with train label list, we need to
  # sort.
  label_list.sort()
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  # save known and unknown data for BERT finetune
  save_ft_data(shuffled_known_train_examples, shuffled_unknown_train_examples,
               FLAGS.few_shot, known_ft_path, unknown_ft_path, label_map,
               max_seq_len, tokenizer, FLAGS.known_num_shots, FLAGS.batch_size)


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.config.set_soft_device_placement(True)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  train_input_fn = None
  ft_known_train_file = None
  train_file = None
  if FLAGS.do_train:
    current_seed = 0
    num_known_classes = FLAGS.num_domains * FLAGS.num_labels_per_domain
    data_output_dir = FLAGS.data_output_dir
    if not tf.gfile.Exists(data_output_dir):
      tf.gfile.MakeDirs(data_output_dir)
    known_ft_path = os.path.join(data_output_dir, "known_ft_train.tf_record")
    unknown_ft_path = os.path.join(data_output_dir,
                                   "unknown_ft_train.tf_record")
    if not tf.gfile.Glob(known_ft_path):
      preprocess_few_shot_training_data(tokenizer, known_ft_path,
                                        unknown_ft_path, current_seed)

    if FLAGS.continual_learning is None:
      assert False, "Not Implemented"
    elif FLAGS.continual_learning == "pretrain":
      train_file = os.path.join(FLAGS.data_output_dir,
                                "known_ft_train.tf_record")
      num_classes = num_known_classes
      num_train_examples = num_known_classes * FLAGS.known_num_shots
      num_shots_per_class = FLAGS.known_num_shots
    elif FLAGS.continual_learning == "few_shot":
      train_file = os.path.join(FLAGS.data_output_dir,
                                "unknown_ft_train.tf_record")
      ft_known_train_file = os.path.join(FLAGS.data_output_dir,
                                         "known_ft_train.tf_record")
      num_unknown_classes = NUM_CLASSES - num_known_classes
      num_classes = num_unknown_classes
      num_train_examples = num_unknown_classes * FLAGS.few_shot
      num_shots_per_class = FLAGS.few_shot

    tpu_split = FLAGS.tpu_split if FLAGS.use_tpu else 1
    if num_shots_per_class < tpu_split:
      steps_per_epoch = 1
    else:
      steps_per_epoch = num_shots_per_class // tpu_split
    num_train_steps = int(steps_per_epoch * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    FLAGS.num_train_steps = num_train_steps
    FLAGS.save_checkpoints_steps = int(steps_per_epoch * FLAGS.save_every_epoch)

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
        ft_known_train_file=ft_known_train_file,
        use_tpu=FLAGS.use_tpu)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  FLAGS.do_eval = False
  eval_input_fn = None
  params = _get_hparams()
  params.update(num_train_steps=num_train_steps)
  if not FLAGS.do_train:
    train_input_fn = eval_input_fn

  experiment_utils.run_experiment(
      model_fn=model_fn,
      train_input_fn=train_input_fn,
      eval_input_fn=train_input_fn,
      params=params)

if __name__ == "__main__":
  app.run(main)
