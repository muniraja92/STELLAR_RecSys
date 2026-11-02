# STELLAR-KG

Code and results for *STELLAR-KG: Spectral Graph Wavelets on Collaborative Knowledge Graphs with Temporal Positional Reasoning for Sequential Recommendation* (P. Muniraja and S. M. Satapathy, revised manuscript submitted to *Knowledge and Information Systems*).

The repository contains the complete pipeline of the revised study on MovieLens-1M: data preparation and knowledge-graph construction, STELLAR-KG (KG-SGW, TPR with PCL and CIR, and the HGF fusion gate) with all ablation variants, the ten baselines, the tuning and multi-seed driver, the statistical and stratified analyses and the scripts that produce every table and figure.

## Layout

| Path | Contents |
|---|---|
| `src/prep_data.py` | Downloads and preprocesses MovieLens-1M (with the RippleNet Satori triples), Amazon Beauty and Amazon Toys and Games (2014), builds the chronological leave-one-out splits and the item knowledge graphs |
| `src/common.py` | Dataset loading, leakage checks, full-catalogue evaluation with pessimistic ties, beyond-accuracy metrics |
| `src/models_cf.py` | Popularity, ItemKNN, EASE, BPR-MF, LightGCN |
| `src/models_kg.py` | KG-SGW (Chebyshev heat-kernel wavelets on the relation-weighted CKG), KGAT, KGIN |
| `src/models_seq.py` | SASRec, BERT4Rec, BSARec and TPR (knowledge tokens, RoPE, time-gap decay, PCL, CIR) |
| `src/fusion.py` | Hierarchical gated fusion (zero-initialised gate fitted on validation targets) |
| `src/run.py` | Runs one model, configuration and seed |
| `src/gdriver.py` | Dependency-aware scheduler: validation grid, configuration selection, seeds 42, 1 and 7 |
| `src/export_peruser.py` | Collects per-user test metrics of all MovieLens-1M runs into `results/strat/ml1m_peruser.npz` |
| `src/stratify.py` | Stratified analysis: per-stratum paired tests, stratified signed-rank tests, Cochran's Q, balanced-population estimates, stratified-bootstrap ranking stability, knowledge-stratum adjustment |
| `src/report.py` | Aggregates runs; paired Wilcoxon tests with Holm correction, bootstrap intervals, effect sizes, group analyses, learned-parameter diagnostics |
| `paper_scripts/` | Produce the LaTeX tables and the figures of the paper from `results/tables` |
| `results/tables/` | Aggregated MovieLens-1M results: per-model means and standard deviations over seeds (`ml1m_main.csv`) and the full report with paired statistics and diagnostics (`ml1m_report.json`) |
| `results/strat/` | Per-user test metrics and the output of the stratified analysis |
| `results/supplementary_amazon/` | Supplementary runs on Amazon Beauty and Toys and Games with a metadata knowledge graph of three relations (brand, category path, price band); not part of the paper's evidence, kept for transparency |

## Reproducing the experiments

```bash
pip install -r requirements.txt
cd src
python prep_data.py                          # downloads raw data into ../raw and writes ../data/*.pkl
python gdriver.py 8 ml1m 42,1,7             # tunes, selects and runs all models (8 parallel jobs)
python report.py ml1m                        # writes ../results/tables
python export_peruser.py && python stratify.py   # stratified analysis; needs MovieLens-1M users.dat
cd ../paper_scripts
python make_tables.py && python fig_results.py && python fig_kernels.py
```

A single run can be started with `python run.py --dataset ml1m --model tpr --seed 42 --cfg '{"drop": 0.2}'`. The original experiments ran on a 32-core CPU server without a GPU, with two threads per run; self-attentive models use a three-hour training budget. The raw-data location can be changed with the `STELLAR_RAW` environment variable.

## Data sources

MovieLens-1M is distributed by GroupLens (https://grouplens.org/datasets/movielens/1m/). The Satori knowledge-graph triples are those released with RippleNet (Wang et al., CIKM 2018). The gender and age groups used for stratification come from `users.dat` of the same release, which is not redistributed here. The supplementary Amazon 2014 files are those of McAuley et al. (SIGIR 2015) and He and McAuley (WWW 2016). Please respect the licences of the original datasets.

## Files from the first submission

The notebook `STELLAR_KG.ipynb` and the `Figure*.pdf` files at the top level belong to the first version of the manuscript and are kept for the record only. That version was affected by the problems described in the response to the reviewers (test-target leakage, an unused knowledge graph, a single-hop approximation of the wavelet operator); none of its results are used in the revised paper. Everything in `src/`, `paper_scripts/` and `results/` belongs to the revised study.
