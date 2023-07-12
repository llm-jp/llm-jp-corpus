# LLM-jp Corpus

This repository contains a data recipe for the LLM-jp corpus.

| Dataset        | Raw Example Count | Token Count     |
|----------------|-------------------|-----------------|
| mC4 (ja)       | 87,425,304        | 159,251,477,726 |
| Wikipedia (ja) | 1,364,746         | 1,761,540,077   |
| Pile (en)      | TBU               | TBU             |
| Wikipedia (en) | 6,630,651         | 5,853,178,749   |
| Stack (code)   | TBU               | TBU             |

## Data Preparation

In `scripts`, we provide scripts to download, filter, and tokenize the data.

## License

The code in this repository is licensed under the Apache 2.0 license.

As for the dataset itself, refer to the licenses of the data subsets:
- [Wikipedia License](https://huggingface.co/datasets/wikipedia#licensing-information)
- [mC4 license](https://huggingface.co/datasets/mc4#licensing-information)
- [Pile License](https://huggingface.co/datasets/EleutherAI/pile#licensing-information)
- [Stack license](https://huggingface.co/datasets/bigcode/the-stack#licensing-information)
