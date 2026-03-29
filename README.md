# Predictive and Preference-Aware Compiler Optimization Using Feature-Based Machine Learning

## M.Tech Project — NIT Tiruchirappalli
**Student:** Satyam Singh Kumre  
**Guide:** Dr. Priyanka Panigrahi  
**Department:** Computer Science and Engineering

---

## 📋 Overview

This project develops a **predictive scheduling framework** that uses Machine Learning to predict the optimal LLVM optimization pass sequence based on static program features and user-supplied performance priorities. Unlike traditional approaches, our system:

- Extracts **66-dimensional features** (56 Autophase + 10 CFG metrics) from LLVM IR
- Accepts a **Metric Priority Vector** (speed/size/compile-time weights)
- Uses an **Ensemble of RF + XGBoost + DNN** to predict the best pass sequence
- Achieves results **competitive with -O3** with **zero iterative compilation overhead**

## 🏗️ Architecture

```
                    ┌──────────────┐
                    │ C/C++ Source  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   LLVM IR    │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼──────┐ ┌──▼──────┐ ┌──▼────────────┐
     │ Autophase (56) │ │CFG (10) │ │Priority Vec(3)│
     └────────┬──────┘ └──┬──────┘ └──┬────────────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │  69-dim Input │
                    └──────┬───────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
       ┌──────▼─────┐ ┌───▼───┐ ┌──────▼──────┐
       │ Random      │ │XGBoost│ │    DNN      │
       │ Forest      │ │       │ │ (PyTorch)   │
       └──────┬──────┘ └───┬───┘ └──────┬──────┘
              │            │             │
              └────────────┼─────────────┘
                           │
                    ┌──────▼───────┐
                    │   Ensemble   │
                    │ (Weighted)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Optimal Pass │
                    │   Sequence   │
                    └──────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
cd "SEM 2 Project"
pip install -r requirements.txt
```

### Run Experiment

```bash
# Complete experiment (dataset + training + evaluation + visualization)
python scripts/run_experiment.py

# Or run individual steps:
python scripts/generate_dataset.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## 📊 Results

After running the experiment, find results in:
- **`results/figures/`** — All plots (speedup, radar, confusion matrices, etc.)
- **`results/tables/`** — CSV comparison tables
- **`results/experiment_summary.json`** — Full experiment summary

## 📁 Project Structure

```
SEM 2 Project/
├── config/config.yaml          # Global configuration
├── src/
│   ├── feature_extraction/     # Autophase + CFG feature extraction
│   ├── priority_vector/        # Metric Priority Vector system
│   ├── models/                 # RF, XGBoost, DNN, Ensemble
│   ├── optimization/           # LLVM pass management
│   ├── baselines/              # -O0/O1/O2/O3, iterative, base paper
│   └── utils/                  # Data loader, logger
├── evaluation/                 # Metrics, comparator, stat tests
├── visualization/              # Performance, feature, priority plots
├── scripts/
│   ├── generate_dataset.py     # Dataset generation
│   └── run_experiment.py       # End-to-end pipeline
├── results/                    # Output directory
└── tests/                      # Unit tests
```

## 📚 References

1. Wang, Z. & O'Boyle, M. (2018). "Machine Learning in Compiler Optimization." *Proc. IEEE*, 106(11), 1879-1901.
2. Ashouri, A.H. et al. (2017). "MiCOMP." *ACM TACO*, 14(3), 1-28.
3. Cummins, C. et al. (2017). "End-to-End Deep Learning of Optimization Heuristics." *PACT*.
4. Huang, Q. et al. (2019). "AutoPhase." *MLSys*.
5. POSET-RL (2023). "Phase Ordering for Optimizing Size and Execution Time."
6. MLGO (Google, 2024). "Machine Learning Guided Compiler Optimization."

## 📄 License

This project is for academic purposes (M.Tech thesis) at NIT Tiruchirappalli.
