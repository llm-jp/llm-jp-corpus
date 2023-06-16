# CommonCrawl (en)

Follow these instructions to create the English CC dataset.

## Requirements

- Python 3.9+

## Installation

```bash
pip install -r requirements.txt
```

## Download

```bash
mkdir -p data  # or create a corresponding symlink
cd data/
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```

## Format

```bash
mkdir -p logs/c4  # or create a corresponding symlink
python reformat.py \
  --data_dir ./data/c4/en \
  --output_dir ./data/c4/processed_en \
  --max_files -1
```
