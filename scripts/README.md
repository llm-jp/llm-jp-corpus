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
python tokenize_data.py --data_dir data/filter/ja_wiki --output_dir data/tokenize/ja_wiki
python tokenize_data.py --data_dir data/filter/en_wiki --output_dir data/tokenize/en_wiki
python tokenize_data.py --data_dir data/filter/ja_cc --output_dir data/tokenize/ja_cc
python tokenize_data.py --data_dir data/filter/en_pile --output_dir data/tokenize/en_pile
python tokenize_data.py --data_dir data/filter/code_stack --output_dir data/tokenize/code_stack
```

## Sampling the data

```bash
mkdir -p data/sample  # or create a corresponding symlink
python sample_data.py --data_dir data/filter/ja_wiki --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/filter/en_wiki --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/filter/ja_cc --output_dir data/sample --token_size -1  # Use all data
python sample_data.py --data_dir data/filter/en_pile --output_dir data/sample --token_size 30000000000  # Use 30B tokens
python sample_data.py --data_dir data/filter/code_stack --output_dir data/sample --token_size 30000000000  # Use 30B tokens
```