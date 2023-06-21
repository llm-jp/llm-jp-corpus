#!/usr/bin/env sh

DATA_DIR=data/
BASE_URL=https://the-eye.eu/public/AI/pile/
wget -P "${DATA_DIR}" "${BASE_URL}/SHA256SUMS.txt"
wget -P "${DATA_DIR}" "${BASE_URL}/test.jsonl.zst"
wget -P "${DATA_DIR}" "${BASE_URL}/val.jsonl.zst"
for i in $(seq -f "%02g" 0 29); do
  mkdir -p "${DATA_DIR}/train"
  wget -P "${DATA_DIR}/train" "$BASE_URL/train/${i}.jsonl.zst"
done
