# STELLAR_RecSys
Spectral-Temporal Enhanced Learning with Latent Attention for Recommendations on Knowledge Graphs


# STELLAR-KG

**Spectral-Temporal Enhanced Learning with Latent Attention for Recommendations on Knowledge Graphs**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

STELLAR-KG is a comprehensive framework for knowledge graph-enhanced recommendations that addresses multi-resolution structural analysis, temporal dynamics, and popularity bias through five novel, synergistic components.

![Architecture](figures/Figure7_Architecture.pdf)

## Key Features

| Component | Description |
|-----------|-------------|
| **SGW** | Spectral Graph Wavelets for multi-resolution graph analysis |
| **PCL** | Prototypical Contrastive Learning with learnable cluster prototypes |
| **CIR** | Causal Invariant Routing for popularity debiasing via IRM |
| **TPR** | Temporal Positional Reasoning with rotary position embeddings |
| **HGF** | Hierarchical Gated Fusion for adaptive score combination |

## Results

Performance on MovieLens-1M benchmark:

| Method | HR@10 | NDCG@10 | MRR@10 | Improvement |
|--------|-------|---------|--------|-------------|
| ItemKNN | 0.5235 | 0.3009 | 0.2328 | — |
| LightGCN | 0.4500 | 0.2433 | 0.1804 | — |
| **STELLAR-KG** | **0.5692** | **0.3267** | **0.2524** | **+8.59%** |

All improvements are statistically significant at p < 0.001.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/STELLAR-KG.git
cd STELLAR-KG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## Dataset

We use the [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) dataset.

| Statistic | Value |
|-----------|-------|
| Users | 2,082 |
| Items | 3,105 |
| Interactions | 202,546 |
| Sparsity | 96.87% |
| KG Entities | 5,205 |
| KG Triples | 437,196 |

## Usage

### Training

```bash
# Train all components
python train.py --config configs/stellar_kg.yaml

# Train individual components
python train.py --component sgw --epochs 30
python train.py --component pcl --epochs 30
python train.py --component cir --epochs 25
python train.py --component tpr --epochs 30
python train.py --component hgf --epochs 100
```

### Evaluation

```bash
# Evaluate STELLAR-KG
python evaluate.py --model stellar_kg --metrics hr ndcg mrr --k 5 10 20
```

### Inference

```python
from stellar_kg import STELLARKG

# Load pretrained model
model = STELLARKG.load_pretrained('checkpoints/stellar_kg.pt')

# Get recommendations for a user
user_id = 42
recommendations = model.recommend(user_id, top_k=10)
print(recommendations)
```

## Project Structure

```
STELLAR-KG/
├── configs/
│   └── stellar_kg.yaml
├── data/
│   └── movielens-1m/
├── src/
│   ├── models/
│   │   ├── sgw.py          # Spectral Graph Wavelets
│   │   ├── pcl.py          # Prototypical Contrastive Learning
│   │   ├── cir.py          # Causal Invariant Routing
│   │   ├── tpr.py          # Temporal Positional Reasoning
│   │   └── hgf.py          # Hierarchical Gated Fusion
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── kg_builder.py
│   │   └── metrics.py
│   └── stellar_kg.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Ablation Study

| Variant | NDCG@10 | Δ Performance |
|---------|---------|---------------|
| STELLAR-KG (Full) | 0.3267 | — |
| w/o TPR | 0.2871 | -12.13% |
| w/o SGW | 0.2982 | -8.74% |
| w/o PCL | 0.3103 | -5.04% |
| w/o CIR | 0.3178 | -2.73% |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{stellar_kg2025,
  title={STELLAR-KG: Spectral-Temporal Enhanced Learning with Latent Attention for Recommendations on Knowledge Graphs},
  author={Scholar, Research and Satapathy, Shashank Mouli},
  journal={},
  year={2025},
  institution={Vellore Institute of Technology}
}
```

## Authors

- **Research Scholar** - Vellore Institute of Technology
- **Dr. Shashank Mouli Satapathy** - Vellore Institute of Technology

## Acknowledgments

This research was supported by the Vellore Institute of Technology.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For questions or issues, please open an issue on GitHub or contact researcher@vit.ac.in.
