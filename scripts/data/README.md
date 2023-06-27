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
python download.py ja_wiki --output_dir data/download/ja_wiki
python download.py en_wiki --output_dir data/download/en_wiki
python download.py ja_cc --output_dir data/download/ja_cc
python download.py en_pile --output_dir data/download/en_pile
python download.py code_stack --output_dir data/download/code_stack
```

## Filtering and reformatting the data

```bash
mkdir -p data/filter  # or create a corresponding symlink
python filter_and_reformat.py ja_wiki --input_dir data/download/ja_wiki --output_dir data/filter/ja_wiki
python filter_and_reformat.py en_wiki --input_dir data/download/en_wiki --output_dir data/filter/en_wiki
python filter_and_reformat.py ja_cc --input_dir data/download/ja_cc --output_dir data/filter/ja_cc
python filter_and_reformat.py en_pile --input_dir data/download/en_pile --output_dir data/filter/en_pile
python filter_and_reformat.py code_stack --input_dir data/download/code_stack --output_dir data/filter/code_stack
```

## Tokenizing the data

```bash
mkdir -p data/tokenize  # or create a corresponding symlink
python tokenize.py ja_wiki --input_dir data/filter/ja_wiki --output_dir data/tokenize/ja_wiki
python tokenize.py en_wiki --input_dir data/filter/en_wiki --output_dir data/tokenize/en_wiki
python tokenize.py ja_cc --input_dir data/filter/ja_cc --output_dir data/tokenize/ja_cc
python tokenize.py en_pile --input_dir data/filter/en_pile --output_dir data/tokenize/en_pile
python tokenize.py code_stack --input_dir data/filter/code_stack --output_dir data/tokenize/code_stack
```