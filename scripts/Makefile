DATA_DIR := data
VERSION := v1.0.1
# ja_wiki, ja_cc, en_wiki, en_pile, code_stack
CORPUS := ja_wiki
SENTENCEPIECE_MODEL := ./code20k_en40k_ja80k.ver2.model
NUM_PROC := 64
TRAIN_TOKEN_SIZE := -1
VALID_TOKEN_SIZE := 10M
EXT := parquet

SHELL := /bin/bash

DOWNLOAD_DIR := $(DATA_DIR)/$(VERSION)/download/$(CORPUS)
FILTER_DIR := $(DATA_DIR)/$(VERSION)/filter/$(CORPUS)
TOKENIZE_DIR := $(DATA_DIR)/$(VERSION)/tokenize/$(CORPUS)
SAMPLE_DIR := $(DATA_DIR)/$(VERSION)/sample/$(CORPUS)

# FILTERED_FILES := $(wildcard $(FILTER_DIR)/*.$(EXT)
# TOKENIZED_FILES := $(patsubst $(FILTER_DIR)/%.$(EXT),$(TOKENIZE_DIR)/%.$(EXT),$(FILTERED_FILES))

.PHONY: tokenize
tokenize:
	python tokenize_data.py \
	--input_path $(FILTER_DIR) \
	--output_dir $(TOKENIZE_DIR) \
	--sentencepiece_model $(SENTENCEPIECE_MODEL) \
	--num_proc $(NUM_PROC) \

.PHONY: filter
filter:
	python filter_data.py \
	$(CORPUS) \
	--data_dir $(DOWNLOAD_DIR) \
	--output_dir $(FILTER_DIR) \

.PHONY: download
download:
	python download_data.py \
	$(CORPUS) \
	--output_dir $(DOWNLOAD_DIR) \

.PHONY: sample
sample:
	python sample_data.py \
	--input_path $(TOKENIZE_DIR) \
	--output_dir $(SAMPLE_DIR) \
	--train_token_size $(TRAIN_TOKEN_SIZE) \
	--valid_token_size $(VALID_TOKEN_SIZE) \
