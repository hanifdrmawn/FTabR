# FTabR: Faster Retrieval-Augmented Deep Learning with Pre-Clustering for Tabular Data

Supplementary code and experiment logs for the paper submitted to *Array* (Elsevier), manuscript ID ARRAY-D-26-00677.

## Structure

```
FTabR/
├── bin/                        # Model code (FTabR variants and baselines)
├── exp/tesis/                  # Experiment results
│   ├── ftabr_v6/               # FTabR ReLU (K=2–50, seeds 0–2)
│   ├── tabr_v3/                # TabR baseline
│   ├── catboost/               # CatBoost baseline
│   ├── xgboost/                # XGBoost baseline
│   ├── lightgbm/               # LightGBM baseline
│   └── tabnet/                 # TabNet baseline
├── exp/sigmoid/
│   ├── tabr_kmeans_v2.16_Sigmoid/  # FTabR Sigmoid K=7 ablation
│   └── tabr_kmeans_v2.15_Sigmoid/  # FTabR Sigmoid K=8 ablation
├── gen_tables_seeds.py         # Generate Tables 2–5 (main results)
├── gen_tables_ablation.py      # Generate Tables 6–9 (ablation)
├── gen_table_inference.py      # Generate Table 4 (inference time)
├── measure_energy.py           # Generate Table 5 (energy in Joules)
├── measure_catboost_inference.py
├── environment.yaml            # Conda environment
└── README.md
```

## Reproducibility

- **Seeds:** {0, 1, 2} for FTabR, TabR, XGBoost, LightGBM, TabNet
- **CatBoost:** single run (checkpoints not retained), marked with † in tables
- **Train/val/test splits:** fixed per dataset, stored in `exp/` config `.toml` files
- **Hardware:** NVIDIA GeForce RTX 3060
- **Energy measurement:** E = P × t, where P = 206.8 W (measured idle GPU draw), t via Python `time` module

## Datasets

| Task           | Datasets |
|----------------|----------|
| Classification | Bank Marketing, Credit, Jannis, Magic Telescope, Phoneme |
| Regression     | Ailerons, Houses, Pol, Superconduct, Wine Quality |

## K-Means Clustering

K-Means is fitted **once, after encoder training**, on the encoded training-set embeddings. It is not updated during training or at inference. Seeding uses K-Means++ with a fixed random seed for reproducibility.

## Notes on Ablation (Tables 6–9)

Tables 6–7 (ReLU) use results from `exp/tesis/ftabr_v6/`.  
Tables 8–9 (Sigmoid) use results from `exp/sigmoid/`.  
All values are **test** accuracy/AUC/F1/RMSE/MAE, mean ± std over seeds {0, 1, 2}.
