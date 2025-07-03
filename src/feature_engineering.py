"""
Feature engineering module for movie recommendation system.
This module provides functions to engineer features from movie data for recommendation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import yaml
import json
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import sentence-transformers, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers available. Will use for semantic embeddings.")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available. Cannot proceed with recommendation system.")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # If no path is provided, construct a default path relative to this file
        # Assumes config.yaml is in `../configs/` relative to `src/`
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, '..', 'configs', 'config.yaml')
        logger.info(f"No config path provided, using default: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_movie_data(movies_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepare movie data for feature engineering
    
    Args:
        movies_df: DataFrame containing movie data
        config: Configuration dictionary
        
    Returns:
        Processed DataFrame
    """
    # Make a copy to avoid modifying the original
    df = movies_df.copy()
    
    # Ensure required columns exist
    required_columns = ['title', 'overview']
    for col in required_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe. Adding empty column.")
            df[col] = ""
    
    # Convert release_date to datetime if it exists
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        # Extract year for potential use in features
        df['release_year'] = df['release_date'].dt.year
    
    return df


def engineer_transformer_features(movies_df: pd.DataFrame, config: Dict[str, Any], 
                                 model_name: str = None) -> Dict[str, Any]:
    """
    Engineer Sentence Transformer features for semantic search
    
    Args:
        movies_df: DataFrame containing movie data
        config: Configuration dictionary
        model_name: Name of the sentence transformer model to use
        
    Returns:
        Dictionary containing embeddings and model
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("Sentence Transformers not available. Cannot engineer transformer features.")
        return {
            'embeddings': None,
            'model': None,
            'processed_df': movies_df
        }
    
    # If model_name is not provided, get it from config
    if model_name is None:
        model_name = config.get('models', {}).get('transformer', {}).get(
            'name', "sentence-transformers/all-MiniLM-L6-v2")
    
    # Make a copy to avoid modifying the original
    df = movies_df.copy()
    
    # Create text representations for each movie
    movie_texts = []
    
    for _, movie in df.iterrows():
        # Start with the title
        text = f"{movie['title']}. "
        
        # Add overview if available
        if 'overview' in movie and pd.notna(movie['overview']):
            text += movie['overview'] + " "
            
        # Add genres if available
        if 'genre_ids' in movie and isinstance(movie['genre_ids'], list):
            # Try to load genre mapping
            genre_mapping = {}
            try:
                # Try to load from a local file first
                genre_file = Path('../notebooks/tmdb_genres.json')
                if genre_file.exists():
                    with open(genre_file, 'r') as f:
                        genres_data = json.load(f)
                        for genre in genres_data.get('genres', []):
                            genre_mapping[genre['id']] = genre['name']
            except Exception:
                # Fallback to hardcoded mapping
                genre_mapping = {
                    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 
                    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
                    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
                    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
                    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
                }
            
            genres = [genre_mapping.get(gid, f"Genre_{gid}") for gid in movie['genre_ids']]
            text += "Genres: " + ", ".join(genres) + ". "
        
        # Add keywords if available
        if 'keywords' in movie and isinstance(movie['keywords'], list):
            text += "Keywords: " + ", ".join(movie['keywords']) + ". "
            
        # Add cast if available
        if 'cast' in movie and isinstance(movie['cast'], list):
            text += "Cast: " + ", ".join(movie['cast']) + "."
            
        movie_texts.append(text)
    
    # Load the Sentence Transformer model
    try:
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        sentence_model = SentenceTransformer(model_name)
        
        # Compute embeddings
        logger.info(f"Computing embeddings for {len(movie_texts)} movies...")
        embeddings = sentence_model.encode(movie_texts, show_progress_bar=True)
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        
        return {
            'embeddings': embeddings,
            'model': sentence_model,
            'processed_df': df,
            'texts': movie_texts
        }
    
    except Exception as e:
        logger.error(f"Failed to create embeddings with Sentence Transformer: {e}")
        return {
            'embeddings': None,
            'model': None,
            'processed_df': df
        } 