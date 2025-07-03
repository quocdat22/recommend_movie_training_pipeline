"""
Main training script for movie recommendation system.
This script finds movies without embeddings, generates them using a Sentence
Transformer model, and stores them in Supabase.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
import time
import mlflow
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
from supabase import create_client, Client
from pathlib import Path

from feature_engineering import (
    load_config,
    prepare_movie_data,
    engineer_transformer_features,
    SENTENCE_TRANSFORMERS_AVAILABLE
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_new_movie_data_from_supabase(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load movie data from a Supabase table for movies that do not have an
    embedding yet (where 'embedding' is NULL).

    Args:
        config: Configuration dictionary.

    Returns:
        DataFrame with new movie data.
    """
    load_dotenv()
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in .env file.")

    supabase: Client = create_client(supabase_url, supabase_key)
    table_name = config.get('data', {}).get('supabase_table', 'movies')

    logger.info(f"Fetching new movies (without embeddings) from Supabase table: {table_name}")

    all_movies = []
    page_size = 1000  # Max limit per request for Supabase
    current_pos = 0

    while True:
        logger.info(f"Fetching new movies from position {current_pos} to {current_pos + page_size - 1}...")
        response = (
            supabase.table(table_name)
            .select('id, title, overview, genre_ids, keywords, top_cast')
            .filter('embedding', 'is', 'null')  # Filter for rows where embedding is not set
            .range(current_pos, current_pos + page_size - 1)
            .execute()
        )

        if response and hasattr(response, 'data'):
            data_chunk = response.data
            all_movies.extend(data_chunk)

            # If the number of returned movies is less than the page size, it's the last page
            if len(data_chunk) < page_size:
                break

            current_pos += page_size
        else:
            # Handle cases with errors or no data on the first page
            break

    if all_movies:
        logger.info(f"Loaded a total of {len(all_movies)} new movies from Supabase.")
        return pd.DataFrame(all_movies)
    else:
        logger.info("No new movies to process were found in Supabase.")
        return pd.DataFrame()

def save_embeddings_to_supabase(movies_df: pd.DataFrame, embeddings: np.ndarray, config: Dict[str, Any]):
    """
    Save movie embeddings to Supabase.

    Args:
        movies_df: DataFrame containing movie IDs and other metadata.
        embeddings: NumPy array of embeddings.
        config: Configuration dictionary.
    """
    load_dotenv()
    supabase_url = os.environ.get('SUPABASE_URL')
    supabase_key = os.environ.get('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found for saving embeddings.")

    supabase: Client = create_client(supabase_url, supabase_key)
    table_name = config.get('data', {}).get('supabase_table', 'movies')

    logger.info(f"Preparing to save {len(embeddings)} embeddings to Supabase table '{table_name}'...")

    # Create a copy of the DataFrame to ensure we don't modify the original
    df_to_upsert = movies_df.copy()

    # Add the embeddings as a new column. Must be a list for JSON serialization.
    df_to_upsert['embedding'] = [emb.tolist() for emb in embeddings]

    # Replace pandas NaN with Python None for JSON compatibility before converting to dict
    df_to_upsert = df_to_upsert.where(pd.notna(df_to_upsert), None)

    # Convert the DataFrame to a list of dictionaries, which is the required format for upsert
    update_data = df_to_upsert.to_dict(orient='records')

    try:
        # Upsert in batches to avoid large request payloads
        batch_size = 100
        for i in range(0, len(update_data), batch_size):
            batch = update_data[i:i + batch_size]
            logger.info(f"Upserting batch {i//batch_size + 1}/{ -(-len(update_data) // batch_size)}...")
            supabase.table(table_name).upsert(batch).execute()

        logger.info("Successfully saved all embeddings to Supabase.")
    except Exception as e:
        logger.error(f"An error occurred while saving embeddings to Supabase: {e}")
        raise

def save_local_artifacts(
    movie_ids: List[Any],
    embeddings: np.ndarray,
    config: Dict[str, Any]
) -> str:
    """
    Save movie IDs and embeddings as local artifacts for the serving API.

    Args:
        movie_ids: List of movie IDs.
        embeddings: NumPy array of embeddings.
        config: Configuration dictionary.

    Returns:
        The absolute path to the output directory.
    """
    # Get the directory of the current script (__file__), which is inside 'src'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get the project root (the parent directory of 'src')
    base_dir = os.path.dirname(script_dir)

    # Get the configured output directory name, defaulting to 'models'
    output_dir_name = config.get('artifacts', {}).get('output_dir', 'models')

    # Construct the full, absolute path for the output directory
    output_dir = os.path.join(base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'movie_embeddings.npy')
    logger.info(f"Saving movie embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)

    # Save movie IDs
    ids_path = os.path.join(output_dir, 'movie_ids.json')
    logger.info(f"Saving movie IDs to {ids_path}")
    with open(ids_path, 'w') as f:
        json.dump(movie_ids, f)

    return output_dir

def main():
    """Main function to run the training process."""
    # ------------------------------------------------------------------
    # Configure MLflow to always write to the central `mlruns` directory
    # located at the project root (one level above this `src` folder).
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent.parent  # e.g., movie-model-training/
    mlruns_dir = base_dir / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")

    # Set the MLflow experiment. If it doesn't exist, MLflow creates it.
    mlflow.set_experiment("Movie Embedding Generation - New Only")

    # Start an MLflow run. All parameters, metrics, and artifacts will be logged to this run.
    with mlflow.start_run():
        start_time = time.time()
        logger.info("Starting movie embedding process for NEW movies only.")

        # Log the start time as a parameter
        mlflow.log_param("run_start_timestamp", pd.to_datetime(start_time, unit='s').isoformat())

        config = load_config()

        # Log key configuration details as parameters
        mlflow.log_param("supabase_table", config.get('data', {}).get('supabase_table', 'movies'))
        model_name_config = config.get('models', {}).get('transformer', {})
        model_name = model_name_config.get('name', "sentence-transformers/all-MiniLM-L6-v2")
        mlflow.log_param("sentence_transformer_model", model_name)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("Sentence Transformers is required but not available. Exiting.")
            mlflow.log_metric("run_status", 0) # Use 0 for failure, 1 for success
            sys.exit(1)

        try:
            # Use the new function to load data
            movies_df = load_new_movie_data_from_supabase(config)
        except Exception as e:
            logger.error(f"Could not load data. Exiting. Error: {e}")
            mlflow.log_metric("run_status", 0)
            sys.exit(1)

        if movies_df.empty:
            logger.info("No new movies to process. Exiting.")
            mlflow.log_metric("num_movies_processed", 0)
            mlflow.log_metric("run_status", 1)
            sys.exit(0)

        num_movies = len(movies_df)
        logger.info(f"Processing {num_movies} new movies.")
        mlflow.log_metric("num_movies_processed", num_movies)

        processed_df = prepare_movie_data(movies_df, config)

        # --- Time and log embedding generation ---
        embedding_start_time = time.time()
        transformer_results = engineer_transformer_features(processed_df, config, model_name)
        embedding_duration = time.time() - embedding_start_time
        mlflow.log_metric("embedding_generation_duration_sec", round(embedding_duration, 2))

        if transformer_results['embeddings'] is None:
            logger.error("Failed to generate embeddings. Exiting.")
            mlflow.log_metric("run_status", 0)
            sys.exit(1)

        # --- Time and log saving to Supabase ---
        save_start_time = time.time()
        save_embeddings_to_supabase(processed_df, transformer_results['embeddings'], config)
        save_duration = time.time() - save_start_time
        mlflow.log_metric("supabase_save_duration_sec", round(save_duration, 2))

        # --- Save local artifacts and log them to MLflow ---
        output_dir = save_local_artifacts(
            movie_ids=processed_df['id'].tolist(),
            embeddings=transformer_results['embeddings'],
            config=config
        )
        mlflow.log_artifacts(output_dir, artifact_path="model_artifacts")

        total_duration = time.time() - start_time
        mlflow.log_metric("total_run_duration_sec", round(total_duration, 2))
        mlflow.log_metric("run_status", 1) # 1 for success
        logger.info(f"Pipeline finished successfully in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()
