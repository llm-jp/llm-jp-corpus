DATA_DIR := data
VERSION := v1.0.1
# ja_wiki, ja_cc, en_wiki, en_pile, code_stack
CORPUS := ja_wiki
SENTENCEPIECE_MODEL := ./code20k_en40k_ja80k.ver2.model
NUM_PROC := 64
# TRAIN_TOKEN_SIZE := -1
# VALID_TOKEN_SIZE := 10M
VALID_EXAMPLES_PER_SHARD := 7K
EXT := parquet

SHELL := /bin/bash

DOWNLOAD_DIR := $(DATA_DIR)/$(VERSION)/download/$(CORPUS)
FILTER_DIR := $(DATA_DIR)/$(VERSION)/filter/$(CORPUS)
TOKENIZE_DIR := $(DATA_DIR)/$(VERSION)/tokenize/$(CORPUS)
SAMPLE_DIR := $(DATA_DIR)/$(VERSION)/sample/$(CORPUS)
SPLIT_DIR := $(DATA_DIR)/$(VERSION)/split/$(CORPUS)

FILTERED_FILES := $(wildcard $(FILTER_DIR)/*.$(EXT))
TOKENIZED_FILES := $(patsubst $(FILTER_DIR)/%.$(EXT),$(TOKENIZE_DIR)/%.$(EXT),$(FILTERED_FILES))
# TOKENIZED_FILES := $(wildcard $(TOKENIZE_DIR)/*.$(EXT))
SPLIT_TRAIN_FILES := $(patsubst $(TOKENIZE_DIR)/%.$(EXT),$(SPLIT_DIR)/%.jsonl,$(TOKENIZED_FILES))
SPLIT_VALIDATION_FILES := $(patsubst $(SPLIT_DIR)/train_%.jsonl,$(SPLIT_DIR)/validation_%.jsonl,$(SPLIT_TRAIN_FILES))

# .PHONY: sample
# sample:
# 	python sample_data.py \
# 	--input_path $(TOKENIZE_DIR) \
# 	--output_dir $(SAMPLE_DIR) \
# 	--train_token_size $(TRAIN_TOKEN_SIZE) \
# 	--valid_token_size $(VALID_TOKEN_SIZE) \

.PHONY: split
split: $(SPLIT_VALIDATION_FILES)

$(SPLIT_VALIDATION_FILES): $(SPLIT_DIR)/validation_%.jsonl: $(TOKENIZE_DIR)/train_%.$(EXT)
	python split_data.py \
	--input_path $< \
	--output_dir $(SPLIT_DIR) \
	--output_format jsonl \
	--valid_examples_per_shard $(VALID_EXAMPLES_PER_SHARD) \
	|| rm $@ $(pathsubst validation,train,$@)

.PHONY: tokenize
tokenize: $(TOKENIZED_FILES)
$(TOKENIZED_FILES): $(TOKENIZE_DIR)/%.$(EXT): $(FILTER_DIR)/%.$(EXT)
	python tokenize_data.py \
	--input_path $< \
	--output_dir $(TOKENIZE_DIR) \
	--sentencepiece_model $(SENTENCEPIECE_MODEL) \
	--num_proc $(NUM_PROC) \

.PHONY: filter
filter:
	python filter_data.py \
	$(CORPUS) \
	--input_dir $(DOWNLOAD_DIR) \
	--output_dir $(FILTER_DIR) \

.PHONY: download
download:
	python download_data.py \
	$(CORPUS) \
	--output_dir $(DOWNLOAD_DIR) \
