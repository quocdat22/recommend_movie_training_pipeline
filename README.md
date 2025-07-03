# Movie Model Training

This module is responsible for building the movie recommendation model's core components. It fetches pre-processed data from Supabase, generates semantic embeddings for movie descriptions using a `sentence-transformers` model, and saves the resulting artifacts.

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- Access to the Supabase database populated by the `movie-data-pipeline`.

### 2. Installation

1.  **Navigate to the directory:**
    ```bash
    cd movie-model-training
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration

Model training parameters and Supabase credentials should be configured in `configs/config.yaml` and a `.env` file respectively.

1.  Create a `.env` file for credentials:
    ```env
    SUPABASE_URL="your_supabase_project_url"
    SUPABASE_KEY="your_supabase_api_key"
    ```
2.  Review and adjust parameters in `configs/config.yaml`.

## Usage

To run the training script:
```bash
python src/train.py
```

## Model Artifacts

The training process will generate the following files:
- `movie_embeddings.npy`: A NumPy array containing the generated sentence embeddings for all movies.
- `movie_ids.json`: A list of movie IDs, where the index of each ID corresponds to its embedding in `movie_embeddings.npy`.

These artifacts are essential for the model serving API to perform semantic similarity searches and provide movie recommendations. They are intended to be uploaded to a cloud storage service like GCS or S3.

# Automated Update Pipeline

The repository provides a convenient shell script that automates the end-to-end process of finding **new** movies without embeddings, generating embeddings, and writing them back to Supabase.

```bash
# From the `movie-model-training` directory
./run_update_pipeline.sh
```

The script will:

1. Activate the virtual environment (if present).
2. Run `src/train.py` to process the new records.
3. Deactivate the environment when finished.

# Experiment Tracking with MLflow

Every training run is logged to MLflow – parameters, metrics, and artifacts (embeddings, IDs).

**Tracking directory**: `movie-model-training/mlruns` (automatically created if missing).

## Launch the MLflow UI

```bash
cd movie-model-training       # project root
mlflow ui                    # starts the UI on http://127.0.0.1:5000

# If you prefer to be explicit
mlflow ui --backend-store-uri ./mlruns
```

Open the URL printed in the terminal to inspect experiments, compare runs, and download artifacts.

# Directory Structure (simplified)

```text
movie-model-training/
├── configs/            # YAML configuration for model & data pipeline
├── mlruns/             # MLflow experiments (auto-generated)
├── models/             # Saved NumPy embeddings & IDs (artifacts)
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code (training, feature engineering, recommend API)
│   ├── train.py
│   ├── feature_engineering.py
│   └── recommend.py
├── run_update_pipeline.sh
└── README.md
```

Feel free to open issues or PRs for questions, improvements, or bug fixes!
