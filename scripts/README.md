# LLM-jp Corpus

## Requirements

- Python 3.9

## Installation

```bash
pip install -r requirements.txt
```

## Downloading the data

```bash
mkdir -p data/download  # or create a corresponding symlink
python download_data.py ja_wiki --output_dir data/download/ja_wiki
python download_data.py en_wiki --output_dir data/download/en_wiki
python download_data.py ja_cc --output_dir data/download/ja_cc
python download_data.py en_pile --output_dir data/download/en_pile
python download_data.py code_stack --output_dir data/download/code_stack
```

## Filtering the data

```bash
mkdir -p data/filter  # or create a corresponding symlink
python filter_data.py ja_wiki --input_dir data/download/ja_wiki --output_dir data/filter/ja_wiki
python filter_data.py en_wiki --input_dir data/download/en_wiki --output_dir data/filter/en_wiki
python filter_data.py ja_cc --input_dir data/download/ja_cc --output_dir data/filter/ja_cc
python filter_data.py en_pile --input_dir data/download/en_pile --output_dir data/filter/en_pile
python filter_data.py code_stack --input_dir data/download/code_stack --output_dir data/filter/code_stack
```

## Tokenizing the data

```bash
mkdir -p data/tokenize  # or create a corresponding symlink
python tokenize_data.py --input_path data/filter/ja_wiki --output_dir data/tokenize/ja_wiki --sentencepiece_model ./spm.model
python tokenize_data.py --input_path data/filter/en_wiki --output_dir data/tokenize/en_wiki --sentencepiece_model ./spm.model
python tokenize_data.py --input_path data/filter/ja_cc --output_dir data/tokenize/ja_cc --sentencepiece_model ./spm.model
python tokenize_data.py --input_path data/filter/en_pile --output_dir data/tokenize/en_pile --sentencepiece_model ./spm.model
python tokenize_data.py --input_path data/filter/code_stack --output_dir data/tokenize/code_stack --sentencepiece_model ./spm.model
```

## Sampling and splitting the data

```bash
mkdir -p data/sample
python sample_data.py --input_path data/tokenize/ja_wiki --output_dir data/sample/ja_wiki --train_token_size -1 --valid_token_size 10M
python sample_data.py --input_path data/tokenize/en_wiki --output_dir data/sample/en_wiki --train_token_size -1 --valid_token_size 10M
python sample_data.py --input_path data/tokenize/ja_cc --output_dir data/sample/ja_cc --train_token_size -1 --valid_token_size 10M
python sample_data.py --input_path data/tokenize/en_pile --output_dir data/sample/en_pile --train_token_size -1 --valid_token_size 10M
python sample_data.py --input_path data/tokenize/code_stack --output_dir data/sample/code_stack --train_token_size -1 --valid_token_size 10M
```

## Extracting validation IDs

```bash
python extract_ids.py ja_wiki --input_path data/sample/ja_wiki/validation_*.jsonl --output_file data/validation_ids/ja_wiki.json
```

## Splitting the data into train and validation sets by validation IDs

```bash
python split_data_by_id.py --input_path data/tokenize/ja_wiki --output_dir data/split/ja_wiki --valid_id_file data/validation_ids/ja_wiki.json --num_proc 6
...
```

---

## Evaluating the filtering quality

```bash
python evaluate_filtering.py ja_cc --input_path benchmark/ja-mc4.valid.labeled.jsonl
```
