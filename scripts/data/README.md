# Data

## Requirements

- Python 3.9

## Downloading the data

```bash
mkdir -p data/download  # or create a corresponding symlink
python download.py ja_wiki --output_dir data/download/ja_wiki
python download.py en_wiki --output_dir data/download/en_wiki
python download.py ja_cc --output_dir data/download/ja_cc
python download.py en_pile --output_dir data/download/en_pile
python download.py code_stack --output_dir data/download/code_stack
```
