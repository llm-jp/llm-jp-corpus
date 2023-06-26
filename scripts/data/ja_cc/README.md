# CommonCrawl (ja)

Follow these instructions to create the Japanese CC dataset.

## Requirements

- Python 3.9+

## Installation

```bash
pip install -r requirements.txt
```

## Download

```bash
mkdir -p data/download  # or create a corresponding symlink
python download.py --output_dir data/download
```

## Reformat

```bash
mkdir -p data/reformat  # or create a corresponding symlink
python reformet.py --data_dir data --output_dir data/reformat
```

## Filtering
```
python sanitize.py in.json out.json
```
