"""
Recommendation module for movie recommendation system.
This module provides functions to recommend movies using embeddings
stored in a pgvector-enabled Supabase database.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence Transformers is available. Will use for text-based search.")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers not available. Text-based search will be disabled.")

# Import local modules
try:
    from feature_engineering import load_config
except ImportError:
    # Add parent directory to path to find feature_engineering module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from feature_engineering import load_config


class MovieRecommender:
    """Movie recommendation system using Supabase pgvector."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the movie recommender.
        
        Args:
            config_path: Optional path to the configuration file.
        """
        self.config = load_config(config_path)
        self.supabase: Client = self._init_supabase_client()
        self.transformer_model: Optional[SentenceTransformer] = self._load_transformer_model()

    def _init_supabase_client(self) -> Client:
        """Initializes and returns the Supabase client."""
        load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Supabase credentials not found in .env file.")
            raise ValueError("Supabase URL and Key must be set.")
        logger.info("Supabase client initialized.")
        return create_client(supabase_url, supabase_key)

    def _load_transformer_model(self) -> Optional[SentenceTransformer]:
        """Loads and returns the Sentence Transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        
        model_name_config = self.config.get('models', {}).get('transformer', {})
        model_name = model_name_config.get('name', "sentence-transformers/all-MiniLM-L6-v2")
        
        try:
            logger.info(f"Loading Sentence Transformer model: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info("Sentence Transformer model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            return None

    def recommend_by_movie_title(
        self, 
        title: str, 
        n_recommendations: int = 10, 
        match_threshold: float = 0.4
    ) -> pd.DataFrame:
        """
        Recommend movies similar to a given movie title by querying Supabase.
        
        Args:
            title: The title of the movie to find recommendations for.
            n_recommendations: The number of recommendations to return.
            match_threshold: The minimum similarity score to consider a match.
            
        Returns:
            A pandas DataFrame with the recommended movies.
        """
        logger.info(f"Finding movie ID for title: '{title}'")
        try:
            movie_response = self.supabase.table("movies").select("id").eq("title", title).limit(1).single().execute()
            movie_id = movie_response.data['id']
            logger.info(f"Found movie ID: {movie_id}")
        except Exception as e:
            logger.error(f"Could not find movie with title '{title}' in the database. Error: {e}")
            return pd.DataFrame()

        logger.info(f"Fetching recommendations for movie ID {movie_id}...")
        try:
            rpc_params = {
                'p_movie_id': movie_id,
                'match_count': n_recommendations,
                'match_threshold': match_threshold
            }
            recommendations_response = self.supabase.rpc("match_movies", rpc_params).execute()
            
            recs_data = recommendations_response.data
            if not recs_data:
                logger.warning("No similar movies found above the threshold.")
                return pd.DataFrame()
                
            logger.info(f"Found {len(recs_data)} recommendations.")
            return pd.DataFrame(recs_data)
        except Exception as e:
            logger.error(f"An error occurred while fetching recommendations: {e}")
            return pd.DataFrame()

    def recommend_by_text(
        self, 
        query: str, 
        n_recommendations: int = 10,
        match_threshold: float = 0.45
    ) -> pd.DataFrame:
        """
        Recommend movies based on a text query using sentence embeddings.
        
        Args:
            query: The text query to search for.
            n_recommendations: The number of recommendations to return.
            match_threshold: The minimum similarity score to consider a match.
            
        Returns:
            A pandas DataFrame with the recommended movies.
        """
        if not self.transformer_model:
            logger.error("Sentence Transformer model is not available. Cannot perform text search.")
            return pd.DataFrame()

        logger.info(f"Generating embedding for query: '{query}'")
        query_embedding = self.transformer_model.encode(query)
        embedding_list = query_embedding.tolist()

        logger.info("Fetching recommendations based on text query...")
        try:
            rpc_params = {
                'query_embedding': embedding_list,
                'match_count': n_recommendations,
                'match_threshold': match_threshold
            }
            recommendations_response = self.supabase.rpc("match_movies_by_text_embedding", rpc_params).execute()
            
            recs_data = recommendations_response.data
            if not recs_data:
                logger.warning("No similar movies found for the query above the threshold.")
                return pd.DataFrame()

            logger.info(f"Found {len(recs_data)} recommendations.")
            return pd.DataFrame(recs_data)
        except Exception as e:
            logger.error(f"An error occurred while fetching recommendations: {e}")
            return pd.DataFrame()


def main():
    """Main function to demonstrate the recommendation system."""
    recommender = MovieRecommender()

    # --- Example 1: Get recommendations for a specific movie ---
    movie_to_test = "Inception" 
    print(f"\n--- Recommendations for '{movie_to_test}' ---")
    movie_recs = recommender.recommend_by_movie_title(movie_to_test, n_recommendations=5)
    if not movie_recs.empty:
        print(movie_recs[['title', 'similarity']].to_string(index=False))
    else:
        print("Could not retrieve recommendations.")

    # --- Example 2: Get recommendations based on a text query ---
    text_query = "A psychological thriller about dreams"
    print(f"\n--- Recommendations for query: '{text_query}' ---")
    text_recs = recommender.recommend_by_text(text_query, n_recommendations=5)
    if not text_recs.empty:
        print(text_recs[['title', 'similarity']].to_string(index=False))
    else:
        print("Could not retrieve recommendations.")

    # --- Example 3: Another text query ---
    text_query_2 = "robots fighting in space"
    print(f"\n--- Recommendations for query: '{text_query_2}' ---")
    text_recs_2 = recommender.recommend_by_text(text_query_2, n_recommendations=5)
    if not text_recs_2.empty:
        print(text_recs_2[['title', 'similarity']].to_string(index=False))
    else:
        print("Could not retrieve recommendations.")

if __name__ == "__main__":
    main() 