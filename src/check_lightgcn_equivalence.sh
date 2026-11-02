#!/usr/bin/env bash
# Implementation check of Section 3.3: KG-SGW with the fixed three-layer LightGCN response as its only
# kernel and without item--entity edges must reproduce LightGCN.  Compare the training loss and
# validation NDCG@10 printed at the 5th and 10th epochs of the two runs (identical to four decimals
# in our check, which used the smaller Amazon Toys graph to keep it fast; any prepared dataset works).
set -e
cd "$(dirname "$0")"
DS=${1:-ml1m}
python run.py --dataset "$DS" --model sgw --tag diag_sgw_lgcn --seed 42 --force \
  --cfg '{"lr": 0.005, "kernels": "lgcn", "use_kg": false, "epochs": 10}'
python run.py --dataset "$DS" --model lightgcn --tag diag_lightgcn --seed 42 --force \
  --cfg '{"lr": 0.005, "epochs": 10}'
