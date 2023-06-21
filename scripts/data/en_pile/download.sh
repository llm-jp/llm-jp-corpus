#!/usr/bin/env sh

DATA_DIR=data/
BASE_URL=https://the-eye.eu/public/AI/pile/
wget -P "${DATA_DIR}" "${BASE_URL}/checksums.sha256"
wget -P "${DATA_DIR}" "${BASE_URL}/test.jsonl.zst"
wget -P "${DATA_DIR}" "${BASE_URL}/val.jsonl.zst"
for i in $(seq -f "%02g" 0 29); do
    wget -P "${DATA_DIR} $BASE_URL/train_${i}.jsonl.zst"
done
