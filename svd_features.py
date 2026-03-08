"""
Latent Graph Feature Extraction Module (SVD).

This module extracts latent features from the compatibility interaction graph:
- Constructs a sparse matrix from user-user interactions
- Applies TruncatedSVD (Matrix Factorization) to learn latent user embeddings
- These embeddings capture implicit "popularity" and "compatibility clusters"
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import pickle
import os
from typing import Dict, Tuple

class GraphSVDModel:
    """Extracts latent features using Matrix Factorization (SVD)."""
    
    MODEL_DIR = 'models'
    
    def __init__(self, n_components: int = 32, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.user_embeddings = None
        
    def fit(self, target_df: pd.DataFrame) -> 'GraphSVDModel':
        """
        Fit SVD on the interaction matrix.
        
        Args:
            target_df: DataFrame with src_user_id, dst_user_id, compatibility_score
        """
        print(f"Fitting SVD (n_components={self.n_components}) on {len(target_df)} interactions...")
        
        # 1. Map all user IDs to matrix indices
        unique_users = set(target_df['src_user_id'].unique()) | set(target_df['dst_user_id'].unique())
        self.user_to_idx = {uid: i for i, uid in enumerate(sorted(unique_users))}
        self.idx_to_user = {i: uid for uid, i in self.user_to_idx.items()}
        
        n_users = len(unique_users)
        print(f"Total unique users in graph: {n_users}")
        
        # 2. Construct Sparse Matrix (Users x Users)
        # We treat this as a directed graph where A->B has score S
        rows = [self.user_to_idx[uid] for uid in target_df['src_user_id']]
        cols = [self.user_to_idx[uid] for uid in target_df['dst_user_id']]
        data = target_df['compatibility_score'].values
        
        interaction_matrix = coo_matrix((data, (rows, cols)), shape=(n_users, n_users))
        
        # 3. Fit TruncatedSVD
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        # We transform the matrix to get user embeddings
        # For a symmetric recommendation task, row embeddings are usually usually sufficient
        # representing the user's "latent preferences"
        self.user_embeddings = self.svd.fit_transform(interaction_matrix)
        
        print(f"SVD Explained Variance Ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        
        return self
    
    def get_embedding(self, user_id: int) -> np.ndarray:
        """Get latent embedding for a user."""
        idx = self.user_to_idx.get(user_id)
        if idx is None:
            # Cold start: return zero vector
            return np.zeros(self.n_components, dtype=np.float32)
        return self.user_embeddings[idx]
    
    def get_all_embeddings(self) -> Dict[int, np.ndarray]:
        """Get dict of all user embeddings."""
        return {uid: self.user_embeddings[idx] for uid, idx in self.user_to_idx.items()}
    
    def save(self, path: str = None) -> None:
        """Save SVD model and embeddings."""
        if path is None:
            path = os.path.join(self.MODEL_DIR, 'graph_svd.pkl')
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'svd': self.svd,
                'user_to_idx': self.user_to_idx,
                'user_embeddings': self.user_embeddings,
                'n_components': self.n_components
            }, f)
        print(f"Saved GraphSVDModel to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'GraphSVDModel':
        """Load SVD model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            
        model = cls(n_components=data['n_components'])
        model.svd = data['svd']
        model.user_to_idx = data['user_to_idx']
        model.user_embeddings = data['user_embeddings']
        
        print(f"Loaded GraphSVDModel from {path}")
        return model

if __name__ == '__main__':
    # Test
    from feature_engineering import load_data
    _, _, target_df = load_data()
    
    svd = GraphSVDModel(n_components=32)
    svd.fit(target_df)
    svd.save()
