# Intellisenz
Intellisenz/
│
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── configs/
│   ├── base.yaml
│   ├── development.yaml
│   └── production.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_time_alignment.ipynb
│   ├── 04_feature_analysis.ipynb
│   ├── 05_model_experiments.ipynb
│   └── 06_model_evaluation.ipynb
│
├── experiments/
│   ├── 001_isolation_forest/
│   ├── 002_lstm_autoencoder/
│   ├── 003_gru_forecaster/
│   ├── 004_matrix_profile/
│   └── 005_model_comparison/
│
├── src/
│   └── intellisenz/
│       │
│       ├── __init__.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   ├── parsing.py
│       │   ├── validation.py
│       │   ├── time_alignment.py
│       │   └── resampling.py
│       │
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── cleaning.py
│       │   ├── missing_values.py
│       │   ├── outliers.py
│       │   ├── scaling.py
│       │   └── pipeline.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── statistical.py
│       │   ├── temporal.py
│       │   ├── vibration.py
│       │   └── windowing.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── isolation_forest.py
│       │   ├── lstm_autoencoder.py
│       │   ├── gru_forecaster.py
│       │   ├── matrix_profile.py
│       │   └── gmm.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── train_isolation_forest.py
│       │   ├── train_lstm.py
│       │   └── train_gru.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── anomaly_metrics.py
│       │   ├── model_comparison.py
│       │   └── plots.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── predict.py
│       │   └── anomaly_detection.py
│       │
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── data_drift.py
│       │   ├── model_drift.py
│       │   └── performance.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── helpers.py
│
├── tests/
│   ├── unit/
│   │   ├── test_parsing.py
│   │   ├── test_preprocessing.py
│   │   ├── test_features.py
│   │   └── test_models.py
│   │
│   └── integration/
│       ├── test_database.py
│       └── test_pipeline.py
│
├── scripts/
│   ├── ingest_data.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_inference.py
│
├── models/
│   ├── development/
│   └── production/
│
├── artifacts/
│   ├── plots/
│   ├── metrics/
│   └── reports/
│
├── mlruns/
│
└── deployment/
    ├── api/
    │   ├── main.py
    │   └── schemas.py
    │
    └── docker/
        └── Dockerfile