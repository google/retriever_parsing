# Few-shot Slot Filling and Intent Classification with Retrieved Examples

This repository contains code for running CLINC and SNIPS experiments in the
paper "Few-shot Slot Filling and Intent Classification with Retrieved Examples"
(to appear in NAACL 2021).

To run the code, you need to first make sure your python version is below 3.8
because the code uses Tensorflow 1.x which is not supported in python 3.8. To
check your python version, run

```bash
python --version
```

If you have a python vesion >= 3.8, you may want to create a new enviroment with
a lower version of python, e.g. using `conda`.

Next install all the required packages and check out this repo.

```bash
cd /my/path/
pip install absl-py tensorflow==1.15 bert-tensorflow
pip install --no-deps -e \
  git://github.com/google-research/language.git#egg=language
git clone https://github.com/google/retriever_parsing
```

Choose and download a BERT model from
[here](https://github.com/google-research/bert#bert) and unzip models files at
`/my/path/bert`.

## CLINC Experiments

First, download CLINC data.

```bash
cd /my/path/
git clone https://github.com/clinc/oos-eval.git
```

The data will be at `/my/path/oos-eval/data/data_full.json`.

```bash
cd /my/path/retriever_parsing
python clinc_similarity_train.py \
 --data_dir=/my/path/oos-eval/data \
 --data_output_dir=/my/path/oos-eval/preprocessed \
 --bert_config_file=/my/path/bert/bert_config.json \
 --vocab_file=/my/path/bert/vocab.txt \
 --init_checkpoint=/my/path/bert/bert_model.ckpt \
 --use_tpu=false
```

## SNIPS Experiments

First, download SNIPS data from
[here](https://atmahou.github.io/attachments/ACL2020data.zip) and unpack it in
`/my/path/snips_data/`. Assume the data is in `/my/path/snips_data/ACL2020data`.

Then preprocess data for each domain, e.g. `AddToPlaylist`.

```bash
python snips_preprocess_data.py \
  --input_dir=/my/path/snips_data/ACL2020data \
  --output_dir=/my/path/snips_data/ACL2020data/preprocessed \
  --target_domain=AddToPlaylist \
  --few_shot=5 \
  --vocab_file=/my/path/bert/vocab.txt
```

To start training

```bash
python snips_similarity_train.py \
 --data_dir=/my/path/snips_data/ACL2020data/preprocessed/5 \
 --few_shot=5 \
 --bert_config_file=/my/path/bert/bert_config.json \
 --vocab_file=/my/path/bert/vocab.txt \
 --init_checkpoint=/my/path/bert/bert_model.ckpt \
 --target_domain=AddToPlaylist \
 --use_tpu=false
```
