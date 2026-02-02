"""
Embedding Generation Module for Pairwise Compatibility Prediction.

This module handles semantic embedding generation and caching:
- Uses all-mpnet-base-v2 for 768-dim embeddings
- Caches embeddings to .npy files for fast reloads
- Generates embeddings for Business_Objectives and Constraints
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional
import json

# Lazy import for sentence_transformers to speed up module loading
_model = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("Loading all-mpnet-base-v2 model...")
        _model = SentenceTransformer('all-mpnet-base-v2')
        print("Model loaded successfully!")
    return _model


class EmbeddingGenerator:
    """Generates and caches semantic embeddings for text features."""
    
    EMBEDDING_DIM = 768
    CACHE_DIR = 'embeddings_cache'
    DEFAULT_MODEL = 'all-mpnet-base-v2'
    
    def __init__(self, cache_dir: str = None, model_name: str = None):
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.model_name = model_name or self.DEFAULT_MODEL
        os.makedirs(self.cache_dir, exist_ok=True)
        self._model = None
    
    @property
    def model(self):
        """Lazy load model only when needed."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        return self._model
    
    def _get_cache_paths(self, dataset_name: str, field: str) -> Tuple[str, str]:
        """Get paths for embedding and ID mapping cache files."""
        emb_path = os.path.join(self.cache_dir, f'{field}_{dataset_name}.npy')
        ids_path = os.path.join(self.cache_dir, f'{field}_{dataset_name}_ids.json')
        return emb_path, ids_path
    
    def _cache_exists(self, dataset_name: str, field: str) -> bool:
        """Check if cache files exist for given dataset and field."""
        emb_path, ids_path = self._get_cache_paths(dataset_name, field)
        return os.path.exists(emb_path) and os.path.exists(ids_path)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for encoding
        
        Returns:
            numpy array of shape (len(texts), 768)
        """
        # Handle empty strings gracefully
        processed_texts = [t if t and str(t).strip() else "No information provided" for t in texts]
        
        print(f"Generating embeddings for {len(processed_texts)} texts...")
        embeddings = self.model.encode(
            processed_texts, 
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.astype(np.float32)
    
    def save_embeddings(self, embeddings: np.ndarray, profile_ids: List[int], 
                        dataset_name: str, field: str) -> None:
        """
        Save embeddings and ID mapping to cache.
        
        Args:
            embeddings: numpy array of shape (n_profiles, 768)
            profile_ids: List of Profile_IDs corresponding to each row
            dataset_name: 'train' or 'test'
            field: 'objectives' or 'constraints'
        """
        emb_path, ids_path = self._get_cache_paths(dataset_name, field)
        
        np.save(emb_path, embeddings)
        with open(ids_path, 'w') as f:
            json.dump(profile_ids, f)
        
        print(f"Saved {field} embeddings for {dataset_name}: {embeddings.shape}")
    
    def load_embeddings(self, dataset_name: str, field: str) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Load embeddings and create ID-to-index mapping.
        
        Args:
            dataset_name: 'train' or 'test'
            field: 'objectives' or 'constraints'
        
        Returns:
            Tuple of (embeddings array, id_to_idx dict)
        """
        emb_path, ids_path = self._get_cache_paths(dataset_name, field)
        
        embeddings = np.load(emb_path)
        with open(ids_path, 'r') as f:
            profile_ids = json.load(f)
        
        id_to_idx = {pid: idx for idx, pid in enumerate(profile_ids)}
        
        print(f"Loaded {field} embeddings for {dataset_name}: {embeddings.shape}")
        return embeddings, id_to_idx
    
    def get_embeddings_with_cache(self, df: pd.DataFrame, dataset_name: str, 
                                   field: str, text_column: str,
                                   force_regenerate: bool = False) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Get embeddings, using cache if available.
        
        Args:
            df: DataFrame with Profile_ID and text column
            dataset_name: 'train' or 'test'
            field: 'objectives' or 'constraints'
            text_column: Name of text column to embed
            force_regenerate: If True, regenerate even if cache exists
        
        Returns:
            Tuple of (embeddings array, id_to_idx dict)
        """
        if not force_regenerate and self._cache_exists(dataset_name, field):
            print(f"Loading cached {field} embeddings for {dataset_name}...")
            return self.load_embeddings(dataset_name, field)
        
        # Generate new embeddings
        profile_ids = df['Profile_ID'].tolist()
        texts = df[text_column].fillna('').tolist()
        
        embeddings = self.generate_embeddings(texts)
        self.save_embeddings(embeddings, profile_ids, dataset_name, field)
        
        id_to_idx = {pid: idx for idx, pid in enumerate(profile_ids)}
        return embeddings, id_to_idx
    
    def generate_all_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                                 force_regenerate: bool = False) -> Dict:
        """
        Generate and cache all embeddings for both datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            force_regenerate: If True, regenerate all embeddings
        
        Returns:
            Dict with all embeddings and mappings
        """
        result = {}
        
        # Train objectives
        emb, id_map = self.get_embeddings_with_cache(
            train_df, 'train', 'objectives', 'Business_Objectives', force_regenerate
        )
        result['train_objectives'] = emb
        result['train_objectives_id_map'] = id_map
        
        # Train constraints
        emb, id_map = self.get_embeddings_with_cache(
            train_df, 'train', 'constraints', 'Constraints', force_regenerate
        )
        result['train_constraints'] = emb
        result['train_constraints_id_map'] = id_map
        
        # Test objectives
        emb, id_map = self.get_embeddings_with_cache(
            test_df, 'test', 'objectives', 'Business_Objectives', force_regenerate
        )
        result['test_objectives'] = emb
        result['test_objectives_id_map'] = id_map
        
        # Test constraints
        emb, id_map = self.get_embeddings_with_cache(
            test_df, 'test', 'constraints', 'Constraints', force_regenerate
        )
        result['test_constraints'] = emb
        result['test_constraints_id_map'] = id_map
        
        return result


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


if __name__ == '__main__':
    # Test embedding generation
    from feature_engineering import load_data
    
    train_df, test_df, _ = load_data()
    
    generator = EmbeddingGenerator()
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    print("\nEmbedding summary:")
    for key, value in embeddings.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {len(value)} mappings")
    
    # Test cosine similarity
    obj_emb = embeddings['train_objectives']
    print(f"\nSample cosine similarity (profile 0 vs 1): {cosine_similarity(obj_emb[0], obj_emb[1]):.4f}")
    print(f"Self-similarity (profile 0 vs 0): {cosine_similarity(obj_emb[0], obj_emb[0]):.4f}")
