#!/usr/bin/env bash
# Token-source controls of Section 5.2: TPR with item tokens from KG-SGW trained on the bipartite graph
# (no knowledge edges) and from LightGCN, instead of the knowledge-graph KG-SGW tokens.
# Requires the finished runs of sgw_nokg and lightgcn (their item embeddings) for the same seed.
# The dropout of 0.2 and lambda_PCL = lambda_CIR = 0.1 are the configuration selected for TPR.
set -e
cd "$(dirname "$0")"
for seed in 42 1 7; do
  for src in sgw_nokg lightgcn; do
    python run.py --dataset ml1m --model tpr --tag "tpr_tok_${src}" --seed "$seed" \
      --cfg "{\"drop\": 0.2, \"pcl\": 0.1, \"cir\": 0.1, \"kg_source\": \"${src}\"}"
  done
done
python report.py ml1m
