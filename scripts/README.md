# Data

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

## Filtering and reformatting the data

```bash
mkdir -p data/filter  # or create a corresponding symlink
python filter_and_reformat_data.py ja_wiki --data_dir data/download/ja_wiki --output_dir data/filter/ja_wiki
python filter_and_reformat_data.py en_wiki --data_dir data/download/en_wiki --output_dir data/filter/en_wiki
python filter_and_reformat_data.py ja_cc --data_dir data/download/ja_cc --output_dir data/filter/ja_cc
python filter_and_reformat_data.py en_pile --data_dir data/download/en_pile --output_dir data/filter/en_pile
python filter_and_reformat_data.py code_stack --data_dir data/download/code_stack --output_dir data/filter/code_stack
```

## Tokenizing the data

```bash
mkdir -p data/tokenize  # or create a corresponding symlink
python tokenize_data.py --data_dir data/filter/ja_wiki --output_dir data/tokenize/ja_wiki --sentencepiece_model ./spm.model
python tokenize_data.py --data_dir data/filter/en_wiki --output_dir data/tokenize/en_wiki --sentencepiece_model ./spm.model
python tokenize_data.py --data_dir data/filter/ja_cc --output_dir data/tokenize/ja_cc --sentencepiece_model ./spm.model
python tokenize_data.py --data_dir data/filter/en_pile --output_dir data/tokenize/en_pile --sentencepiece_model ./spm.model
python tokenize_data.py --data_dir data/filter/code_stack --output_dir data/tokenize/code_stack --sentencepiece_model ./spm.model
```

## Sampling the data

```bash
mkdir -p data/sample  # or create a corresponding symlink
python sample_data.py --data_dir data/tokenize/ja_wiki --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/tokenize/en_wiki --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/tokenize/ja_cc --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/tokenize/en_pile --output_dir data/sample --token_size 25000000000  # Use 25B tokens
python sample_data.py --data_dir data/tokenize/code_stack --output_dir data/sample --token_size 10000000000  # Use 10B tokens
```
