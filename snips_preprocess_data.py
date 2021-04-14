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
r"""Splits and preprocesses SNIPS data.

Command example: running in local (recommended)
python snips_preprocess_data.py \
  --input_dir=/my/path/snips_data/ACL2020data \
  --output_dir=/my/path/snips_data/ACL2020data/preprocessed \
  --target_domain=AddToPlaylist \
  --few_shot=5 \
  --vocab_file=/my/path/bert/vocab.txt
"""

import collections
import json
import os
import pickle
import random
from absl import flags
from bert import tokenization
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_dir", None,
    "The input data dir. Should contain folders with json files, e.g. "
    "'xval_snips_shot_5'.")
flags.DEFINE_string("output_dir", None,
                    "The directory with data preprocessing outputs")
flags.DEFINE_string("target_domain", None, "target domain")
flags.DEFINE_integer("few_shot", 5, "number of shots")
flags.DEFINE_integer("max_seq_len", 70, "max sequence length")
flags.DEFINE_integer("random_seed", 42, "random seed")
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")


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
domain2slot = {
    "AddToPlaylist": [
        "music_item", "playlist_owner", "entity_name", "playlist", "artist"
    ],
    "BookRestaurant": [
        "city", "facility", "timeRange", "restaurant_name", "country",
        "cuisine", "restaurant_type", "served_dish", "party_size_number", "poi",
        "sort", "spatial_relation", "state", "party_size_description"
    ],
    "GetWeather": [
        "city", "state", "timeRange", "current_location", "country",
        "spatial_relation", "geographic_poi", "condition_temperature",
        "condition_description"
    ],
    "PlayMusic": [
        "genre", "music_item", "service", "year", "playlist", "album", "sort",
        "track", "artist"
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
all_domains = [
    "AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook",
    "SearchCreativeWork", "SearchScreeningEvent"
]

# from TapNet slot filling dataset
train_val_domain_map = {
    "AddToPlaylist": "RateBook",
    "BookRestaurant": "SearchCreativeWork",
    "GetWeather": "PlayMusic",
    "PlayMusic": "AddToPlaylist",
    "RateBook": "SearchScreeningEvent",
    "SearchCreativeWork": "GetWeather",
    "SearchScreeningEvent": "BookRestaurant"
}
domain_num_to_name = {
    "1": "GetWeather",
    "2": "PlayMusic",
    "3": "AddToPlaylist",
    "4": "RateBook",
    "5": "SearchScreeningEvent",
    "6": "BookRestaurant",
    "7": "SearchCreativeWork"
}
domain_name_to_num = {
    "GetWeather": "1",
    "PlayMusic": "2",
    "AddToPlaylist": "3",
    "RateBook": "4",
    "SearchScreeningEvent": "5",
    "BookRestaurant": "6",
    "SearchCreativeWork": "7"
}

domain_num_utterances = {
    "AddToPlaylist": 2042,
    "BookRestaurant": 2073,
    "GetWeather": 2100,
    "PlayMusic": 2100,
    "RateBook": 2056,
    "SearchCreativeWork": 2054,
    "SearchScreeningEvent": 2059
}


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               guid,
               text_a,
               text_b=None,
               label=None,
               start_i=-1,
               end_i=-1,
               label_seq=None,
               domain=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
      start_i: int. Start token position of the span.
      end_i: int. End token position of the span.
      label_seq: string. BIO tagging sequence.
      domain: string. Domain of the example.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.start_i = start_i
    self.end_i = end_i
    self.label_seq = label_seq
    self.domain = domain


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
               label_ids,
               slot_name_id,
               start_i,
               end_i,
               is_real_example=True,
               real_seq_len=-1,
               domain_id=-1):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.slot_name_id = slot_name_id
    self.start_i = start_i
    self.end_i = end_i
    self.is_real_example = is_real_example
    self.real_seq_len = real_seq_len
    self.domain_id = domain_id


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


def convert_single_example(ex_index,
                           example,
                           label_map,
                           domain_map,
                           max_seq_length,
                           tokenizer,
                           pad_token_label_id=-100):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_ids=[0] * max_seq_length,
        slot_name_id=-1,
        start_i=-1,
        end_i=-1,
        is_real_example=False,
        real_seq_len=-1,
        domain_id=-1)

  tokens_a = []
  label_ids = []
  segment_total_len = 0
  ori_start_i = example.start_i
  ori_end_i = example.end_i

  for w_idx, (word, label) in enumerate(
      zip(example.text_a.split(), example.label_seq.split())):
    # In tapnet preprocessed data, there can be unknown tokens that will be
    # tokenized to empty strings
    word_tokens = tokenizer.tokenize(word)
    if not word_tokens:
      word_tokens = ["[UNK]"]
    tokens_a.extend(word_tokens)
    label_ids.extend([label_map[label]] + [pad_token_label_id] *
                     (len(word_tokens) - 1))
    if w_idx == ori_start_i:
      example.start_i = segment_total_len
    if w_idx == ori_end_i:
      example.end_i = segment_total_len + len(word_tokens) - 1
    segment_total_len += len(word_tokens)
  try:
    assert example.start_i <= example.end_i and example.end_i < len(tokens_a), \
        "start: %s; end: %s; len(tokens_a): %s" % (example.start_i, example.end_i, len(tokens_a))
  except:
    print("%%%" * 30)
    print(example.text_a)
    print(example.label_seq)
    print(example.label)
    print(ori_start_i)
    print(ori_end_i)
    print(tokens_a)
    print(label_ids)
    print(example.start_i)
    print(example.end_i)
    assert False
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

  label_ids = [pad_token_label_id] + label_ids + [pad_token_label_id]
  start_i = example.start_i + 1
  end_i = example.end_i + 1

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  real_seq_len = len(input_ids)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(pad_token_label_id)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length, "len(label_id): %d" % len(label_ids)

  slot_name_id = label_map[example.label]
  domain_id = domain_map[example.domain]
  if ex_index < 5:
    print("*** Example ***")
    print("guid: %s" % (example.guid))
    print("tokens: %s" %
          " ".join([tokenization.printable_text(x) for x in tokens]))
    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    print("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    print("start_i: %s" % start_i)
    print("end_i: %s" % end_i)
    print("slot_name: %s (id = %d)" % (example.label, slot_name_id))
    print("domain_name: %s (id = %d)" % (example.domain, domain_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_ids,
      slot_name_id=slot_name_id,
      start_i=start_i,
      end_i=end_i,
      is_real_example=True,
      real_seq_len=real_seq_len,
      domain_id=domain_id)
  return feature


def file_based_convert_examples_to_features(examples, label_map, domain_map,
                                            max_seq_length, tokenizer,
                                            output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      print("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map, domain_map,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    features["slot_name_id"] = create_int_feature([feature.slot_name_id])
    features["start_i"] = create_int_feature([feature.start_i])
    features["end_i"] = create_int_feature([feature.end_i])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])
    features["real_seq_len"] = create_int_feature([feature.real_seq_len])
    features["domain_id"] = create_int_feature([feature.domain_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def add_utterance_examples(data_list, return_list, mode, is_query=False):
  """Appends a list of InputExample from `data_list` after `return_list`."""
  for u_id, u_line in enumerate(data_list):
    utterance = u_line[0]
    label = u_line[1]
    domain_name = u_line[2]

    in_slot = False
    i = -1
    start = -1
    end = -1
    slot_name = "None"
    query_added = False
    for i, token in enumerate(label.split()):
      if token[0] in ["B", "O"]:
        if in_slot:
          end = i - 1
          assert slot_name != "None"
          assert end >= start and start > -1, "start: %s; end: %s" % (start,
                                                                      end)
          if is_query:
            guid = "%s-q-%s-%s-%s" % (mode, domain_name, u_id, i)
          else:
            guid = "%s-%s-%s-%s" % (mode, domain_name, u_id, i)
          return_list.append(
              InputExample(
                  guid=guid,
                  text_a=utterance,
                  label=slot_name,
                  start_i=start,
                  end_i=end,
                  label_seq=label,
                  domain=domain_name))
          if is_query:
            query_added = True
            break

        if token[0] == "B":
          slot_name = token.split("-")[-1]
          start = i
          in_slot = True
        else:
          start = -1
          end = -1
          in_slot = False
          slot_name = "None"
    if in_slot and i >= 0:
      if is_query:
        guid = "%s-q-%s-%s-%s" % (mode, domain_name, u_id, i)
      else:
        guid = "%s-%s-%s-%s" % (mode, domain_name, u_id, i)
      if not (is_query and query_added):
        return_list.append(
            InputExample(
                guid=guid,
                text_a=utterance,
                label=slot_name,
                start_i=start,
                end_i=i,
                label_seq=label,
                domain=domain_name))
  return return_list


def get_data(input_data_path, mode):
  """Reads data from `input_data_path`."""
  if mode == "src":
    train_examples = list()
    with tf.io.gfile.GFile(input_data_path) as f:
      raw_data = json.load(f)
      for domain_i, domain_info in raw_data.items():
        domain_i_non_repeat_examples = dict()
        for data_sample in domain_info:
          support_info = data_sample["support"]
          query_info = data_sample["batch"]
          # only consider not repeated examples
          for seq_id in range(len(support_info["seq_ins"])):
            seq_input_i_text = " ".join(
                w for w in support_info["seq_ins"][seq_id]
            )  # cannot use list for hashing
            if seq_input_i_text not in domain_i_non_repeat_examples:
              seq_input_i_labels = " ".join(
                  w for w in support_info["seq_outs"][seq_id])
              seq_input_i_domain_name = support_info["labels"][seq_id]
              domain_i_non_repeat_examples[seq_input_i_text] = (
                  seq_input_i_text, seq_input_i_labels, seq_input_i_domain_name)
          for seq_id in range(len(query_info["seq_ins"])):
            seq_input_i_text = " ".join(w for w in query_info["seq_ins"][seq_id]
                                       )  # cannot use list for hashing
            if seq_input_i_text not in domain_i_non_repeat_examples:
              seq_input_i_labels = " ".join(
                  w for w in query_info["seq_outs"][seq_id])
              seq_input_i_domain_name = query_info["labels"][seq_id]
              domain_i_non_repeat_examples[seq_input_i_text] = (
                  seq_input_i_text, seq_input_i_labels, seq_input_i_domain_name)
        print("Domain: %s has %d unique examples" %
              (domain_i, len(domain_i_non_repeat_examples)))

        train_examples = add_utterance_examples(
            domain_i_non_repeat_examples.values(), train_examples, mode)
    return train_examples
  else:
    support_example_num = 0
    query_examples = list()
    support_examples = list()
    support_example_ids = list()
    with tf.io.gfile.GFile(input_data_path) as f:
      raw_data = json.load(f)
      for domain_i, domain_info in raw_data.items():
        for data_sample_id, data_sample in enumerate(domain_info):
          support_info = data_sample["support"]
          query_info = data_sample["batch"]

          sample_support_list = list()
          sample_query_list = list()

          for seq_id in range(len(support_info["seq_ins"])):
            seq_input_i_text = " ".join(
                w for w in support_info["seq_ins"][seq_id]
            )  # cannot use list for hashing
            seq_input_i_labels = " ".join(
                w for w in support_info["seq_outs"][seq_id])
            seq_input_i_domain_name = support_info["labels"][seq_id]
            sample_support_list.append(
                (seq_input_i_text, seq_input_i_labels, seq_input_i_domain_name))
          sample_support_examples = add_utterance_examples(
              sample_support_list, [], mode)
          support_example_ids.append(
              (support_example_num,
               support_example_num + len(sample_support_examples) - 1))
          support_example_num += len(sample_support_examples)
          support_examples.extend(sample_support_examples)

          for seq_id in range(len(query_info["seq_ins"])):
            seq_input_i_text = " ".join(w for w in query_info["seq_ins"][seq_id]
                                       )  # cannot use list for hashing
            seq_input_i_labels = " ".join(
                w for w in query_info["seq_outs"][seq_id])
            seq_input_i_domain_name = query_info["labels"][seq_id]
            sample_query_list.append(
                (seq_input_i_text, seq_input_i_labels, seq_input_i_domain_name))
          sample_query_examples = add_utterance_examples(
              sample_query_list, [], mode, is_query=True)
          query_examples.extend(sample_query_examples)
    return query_examples, support_examples, support_example_ids


def get_label_map():
  """Returns a mapping from label to ID."""
  label_map = dict()
  i = 0
  for name in slot_list:
    label_map[name] = i
    i += 1
  label_map["O"] = i
  i += 1
  for name in slot_list:
    for prefix in ["B-", "I-"]:
      label_map[prefix + name] = i
      i += 1
  print("total number of labels in label_map: %d" % len(label_map))
  assert len(label_map) == len(set(
      label_map.values())), "len(label_map_values) = %d" % len(
          set(label_map.values()))
  return label_map


def group_examples(examples):
  """Groups examples by their slot types."""
  max_num = 0
  examples_by_type = dict()
  for ex in examples:
    if ex.label not in examples_by_type:
      examples_by_type[ex.label] = list()
    examples_by_type[ex.label].append(ex)
  for slot_type in examples_by_type:
    print(slot_type, len(examples_by_type[slot_type]))
    if len(examples_by_type[slot_type]) > max_num:
      max_num = len(examples_by_type[slot_type])
  return max_num, examples_by_type


def process_snips_data(input_dir, src_domain_train_path, src_support_path,
                       tgt_support_path, tgt_domain_test_path,
                       tgt_support_id_path, val_path, val_support_path,
                       val_support_id_path, tokenizer):
  """Processes SNIPS data."""
  file_num = domain_name_to_num[FLAGS.target_domain]
  if FLAGS.few_shot == 1:
    src_input_path = os.path.join(input_dir, "snips_train_%s.json" % file_num)
    val_input_path = os.path.join(input_dir, "snips_valid_%s.json" % file_num)
    test_input_path = os.path.join(input_dir, "snips_test_%s.json" % file_num)
  else:
    src_input_path = os.path.join(input_dir,
                                  "snips-train-%s-shot-5.json" % file_num)
    val_input_path = os.path.join(input_dir,
                                  "snips-valid-%s-shot-5.json" % file_num)
    test_input_path = os.path.join(input_dir,
                                   "snips-test-%s-shot-5.json" % file_num)

  src_train_examples = get_data(src_input_path, mode="src")
  val_examples, val_support_examples, val_support_ids = get_data(
      val_input_path, mode="val")
  test_examples, test_support_examples, test_support_ids = get_data(
      test_input_path, mode="tgt")

  print("src_train_examples: %d" % len(src_train_examples))
  print("val_examples: %d" % len(val_examples))
  print("val_support_examples: %d" % len(val_support_examples))
  print("val_support_ids: %d" % len(val_support_ids))
  print("val_support_ids last element: %s" % str(val_support_ids[-1]))
  print("test_examples: %d" % len(test_examples))
  print("test_support_examples: %d" % len(test_support_examples))
  print("test_support_ids: %d" % len(test_support_ids))
  print("test_support_ids last element: %s" % str(test_support_ids[-1]))

  # Domain: AddToPlaylist;
  # Seen samples: 480; Unseen samples: 1062; Total samples: 1542
  # Domain: BookRestaurant;
  # Seen samples: 40; Unseen samples: 1533; Total samples: 1573
  # Domain: GetWeather;
  # Seen samples: 623; Unseen samples: 977; Total samples: 1600
  # Domain: PlayMusic;
  # Seen samples: 386; Unseen samples: 1214; Total samples: 1600
  # Domain: RateBook;
  # Seen samples: 0; Unseen samples: 1556; Total samples: 1556
  # Domain: SearchCreativeWork;
  # Seen samples: 1554; Unseen samples: 0; Total samples: 1554
  # Domain: SearchScreeningEvent;
  # Seen samples: 168; Unseen samples: 1391; Total samples: 1559

  # group training examples so that examples with the same slot type will be
  # grouped together
  print()
  print("--- grouping src training examples ---")
  max_src_train_num, grouped_src_train_examples = group_examples(
      src_train_examples)
  print("maximum number of slot type in src training example is %d" %
        max_src_train_num)

  support_src_train_examples = list()
  for ex in src_train_examples:
    support_src_train_examples.append(
        InputExample(
            guid=ex.guid,
            text_a=ex.text_a,
            label=ex.label,
            start_i=ex.start_i,
            end_i=ex.end_i,
            label_seq=ex.label_seq,
            domain=ex.domain))

  # In order to use all the training data, we pad utterances for each slot type
  # so that all slot type has the same number of training utterances
  balanced_src_train_examples = list()
  random.seed(FLAGS.random_seed)
  for slot_type in grouped_src_train_examples:
    slot_examples = grouped_src_train_examples[slot_type]
    padded_slot_examples = list()
    padded_slot_examples.extend(slot_examples)
    repeat_i = 0
    while len(padded_slot_examples) < max_src_train_num:
      # Note: need to deep copy examples, otherwise if change one example
      # (for new start_i), other repeated ones will be changed (if simply do
      # shuffle and extend then will only repeat references).
      copy_slot_examples = list()
      for ex in slot_examples:
        copy_slot_examples.append(
            InputExample(
                guid="cp%d-" % repeat_i + ex.guid,
                text_a=ex.text_a,
                label=ex.label,
                start_i=ex.start_i,
                end_i=ex.end_i,
                label_seq=ex.label_seq,
                domain=ex.domain))
      random.shuffle(copy_slot_examples)
      padded_slot_examples.extend(copy_slot_examples)
      repeat_i += 1
    balanced_src_train_examples.extend(padded_slot_examples[:max_src_train_num])
  print("total number of training slot types: %d" %
        len(grouped_src_train_examples.keys()))
  print("total src train examples num: %d" % len(balanced_src_train_examples))
  print()
  print("---" * 20)
  # Domain: AddToPlaylist;
  # src_max: 2521; src_types: 37; tgt_max: 20; tgt_types: 5; src_num: 93277;
  # tgt_num: 100
  # Domain: BookRestaurant;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:
  # Domain: GetWeather;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:
  # Domain: PlayMusic;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:
  # Domain: RateBook;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:
  # Domain: SearchCreativeWork;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:
  # Domain: SearchScreeningEvent;
  # src_max: ; src_types: ; tgt_max: ; tgt_types: ; src_num: ; tgt_num:

  label_map = get_label_map()
  domain_map = {}
  for (i, domain) in enumerate(all_domains):
    domain_map[domain] = i

  max_seq_len = FLAGS.max_seq_len

  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing src training data ***")
  file_based_convert_examples_to_features(balanced_src_train_examples,
                                          label_map, domain_map, max_seq_len,
                                          tokenizer, src_domain_train_path)
  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing src support data ***")
  file_based_convert_examples_to_features(support_src_train_examples, label_map,
                                          domain_map, max_seq_len, tokenizer,
                                          src_support_path)
  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing val query data ***")
  file_based_convert_examples_to_features(val_examples, label_map, domain_map,
                                          max_seq_len, tokenizer, val_path)
  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing val support data ***")
  file_based_convert_examples_to_features(val_support_examples, label_map,
                                          domain_map, max_seq_len, tokenizer,
                                          val_support_path)

  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing test query data ***")
  file_based_convert_examples_to_features(test_examples, label_map, domain_map,
                                          max_seq_len, tokenizer,
                                          tgt_domain_test_path)
  print("====" * 20)
  print("====" * 20)
  print("*** preprocessing test support data ***")
  file_based_convert_examples_to_features(test_support_examples, label_map,
                                          domain_map, max_seq_len, tokenizer,
                                          tgt_support_path)
  print("====" * 20)
  print("====" * 20)
  print("*** saving support ids for val and test ***")
  with tf.gfile.Open(val_support_id_path, "wb") as val_support_id_file:
    pickle.dump(val_support_ids, val_support_id_file)
  with tf.gfile.Open(tgt_support_id_path, "wb") as test_support_id_file:
    pickle.dump(test_support_ids, test_support_id_file)


def main(_):
  input_path_dir = FLAGS.input_dir
  output_path_dir = FLAGS.output_dir

  output_dir = os.path.join(output_path_dir, str(FLAGS.few_shot))
  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  if FLAGS.few_shot == 1:
    input_dir = os.path.join(input_path_dir, "xval_snips")
  elif FLAGS.few_shot == 5:
    input_dir = os.path.join(input_path_dir, "xval_snips_shot_5")
  else:
    assert False, "not avaiable from the tapnet slot filling paper"

  random.seed(FLAGS.random_seed)

  domain_output_dir = os.path.join(output_dir, FLAGS.target_domain)
  if not tf.gfile.Glob(domain_output_dir):
    tf.gfile.MakeDirs(domain_output_dir)

  vocab_file = FLAGS.vocab_file
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=True)

  src_domain_train_path = os.path.join(domain_output_dir, "train.tf_record")
  src_support_path = os.path.join(domain_output_dir, "src_support.tf_record")
  tgt_support_path = os.path.join(domain_output_dir,
                                  "%s_support.tf_record" % FLAGS.target_domain)
  tgt_domain_test_path = os.path.join(domain_output_dir,
                                      "%s_test.tf_record" % FLAGS.target_domain)
  tgt_support_id_path = os.path.join(domain_output_dir, "test_support_id.pkl")
  val_path = os.path.join(domain_output_dir, "val.tf_record")
  val_support_path = os.path.join(domain_output_dir, "val_support.tf_record")
  val_support_id_path = os.path.join(domain_output_dir, "val_support_id.pkl")

  process_snips_data(input_dir, src_domain_train_path, src_support_path,
                     tgt_support_path, tgt_domain_test_path,
                     tgt_support_id_path, val_path, val_support_path,
                     val_support_id_path, tokenizer)


if __name__ == "__main__":
  tf.app.run(main)
