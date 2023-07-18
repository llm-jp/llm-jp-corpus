# LLM-jp Corpus

This repository contains scripts to reproduce the LLM-jp corpus.

| Dataset        | Token Count |
|----------------|-------------|
| mC4 (ja)       | 159B        |
| Wikipedia (ja) | 2B          |
| Pile (en)      | 25B         |
| Wikipedia (en) | 6B          |
| Stack (code)   | 10B         |

## Data Preparation

In `scripts`, we provide scripts to download, filter, and tokenize the data.

## License

The code in this repository is licensed under the Apache 2.0 license.

As for the dataset itself, refer to the licenses of the data subsets:
- [Wikipedia license](https://huggingface.co/datasets/wikipedia#licensing-information)
- [mC4 license](https://huggingface.co/datasets/mc4#licensing-information)
- [Pile license](https://huggingface.co/datasets/EleutherAI/pile#licensing-information)
- [Stack license](https://huggingface.co/datasets/bigcode/the-stack#licensing-information)
