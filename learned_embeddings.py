"""
Learned Categorical Embeddings Module.

Trains embedding layers for categorical features (Industry, Role, Location)
end-to-end with compatibility prediction, so similar categories have similar vectors.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pickle
import os
from typing import Dict, List, Tuple
from tqdm import tqdm


class CategoricalEmbeddingModel(nn.Module):
    """
    Model that learns embeddings for categorical features.
    
    Computes compatibility as function of learned category similarities.
    """
    
    def __init__(self, vocab_sizes: Dict[str, int], embedding_dims: Dict[str, int]):
        """
        Args:
            vocab_sizes: Dict of {feature_name: vocabulary_size}
            embedding_dims: Dict of {feature_name: embedding_dimension}
        """
        super().__init__()
        
        self.feature_names = list(vocab_sizes.keys())
        
        # Create embedding layers
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], embedding_dims[name])
            for name in self.feature_names
        })
        
        # Compute total embedding dim (2 * each dim for src and dst)
        total_dim = sum(embedding_dims.values()) * 2
        
        # Also add similarity features
        n_similarity_features = len(self.feature_names)  # One cosine per category
        
        # MLP head
        self.head = nn.Sequential(
            nn.Linear(total_dim + n_similarity_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, src_cats: Dict[str, torch.Tensor], dst_cats: Dict[str, torch.Tensor]):
        """
        Forward pass.
        
        Args:
            src_cats: Dict of {feature_name: tensor of indices for src}
            dst_cats: Dict of {feature_name: tensor of indices for dst}
        """
        src_embs = []
        dst_embs = []
        similarities = []
        
        for name in self.feature_names:
            src_emb = self.embeddings[name](src_cats[name])
            dst_emb = self.embeddings[name](dst_cats[name])
            
            src_embs.append(src_emb)
            dst_embs.append(dst_emb)
            
            # Cosine similarity for this category
            cos_sim = nn.functional.cosine_similarity(src_emb, dst_emb, dim=1)
            similarities.append(cos_sim.unsqueeze(1))
        
        # Concatenate all embeddings
        src_concat = torch.cat(src_embs, dim=1)
        dst_concat = torch.cat(dst_embs, dim=1)
        sim_concat = torch.cat(similarities, dim=1)
        
        # Full feature vector
        features = torch.cat([src_concat, dst_concat, sim_concat], dim=1)
        
        return self.head(features).squeeze()
    
    def get_embeddings(self, name: str) -> np.ndarray:
        """Get learned embeddings for a category."""
        return self.embeddings[name].weight.detach().cpu().numpy()


class LearnedCategoricalEncoder:
    """
    Encoder that learns categorical embeddings from compatibility data.
    """
    
    CATEGORICAL_COLS = ['Industry', 'Role', 'Location_City', 'Company_Name']
    EMBEDDING_DIMS = {
        'Industry': 8,
        'Role': 8,
        'Location_City': 8,
        'Company_Name': 4
    }
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.label_encoders = {}
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def fit(self, train_df: pd.DataFrame, target_df: pd.DataFrame,
            epochs: int = 50, batch_size: int = 512, lr: float = 0.01):
        """
        Train the categorical embedding model.
        
        Args:
            train_df: DataFrame with profile features
            target_df: DataFrame with compatibility scores
        """
        print("\nTraining Learned Categorical Embeddings")
        print(f"Device: {self.device}")
        
        # Fit label encoders
        vocab_sizes = {}
        for col in self.CATEGORICAL_COLS:
            le = LabelEncoder()
            train_df[f'{col}_encoded'] = le.fit_transform(train_df[col].fillna('UNKNOWN'))
            self.label_encoders[col] = le
            vocab_sizes[col] = len(le.classes_)
            print(f"  {col}: {vocab_sizes[col]} categories")
        
        # Create id -> encoded mapping
        id_to_encoded = {}
        for _, row in train_df.iterrows():
            id_to_encoded[row['Profile_ID']] = {
                col: row[f'{col}_encoded'] for col in self.CATEGORICAL_COLS
            }
        
        # Prepare training data
        src_data = {col: [] for col in self.CATEGORICAL_COLS}
        dst_data = {col: [] for col in self.CATEGORICAL_COLS}
        scores = []
        
        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="Preparing data"):
            src_id = row['src_user_id']
            dst_id = row['dst_user_id']
            
            if src_id in id_to_encoded and dst_id in id_to_encoded:
                for col in self.CATEGORICAL_COLS:
                    src_data[col].append(id_to_encoded[src_id][col])
                    dst_data[col].append(id_to_encoded[dst_id][col])
                scores.append(row['compatibility_score'])
        
        # Convert to tensors
        src_tensors = {col: torch.LongTensor(src_data[col]) for col in self.CATEGORICAL_COLS}
        dst_tensors = {col: torch.LongTensor(dst_data[col]) for col in self.CATEGORICAL_COLS}
        y_tensor = torch.FloatTensor(scores)
        
        # Split train/val
        n = len(scores)
        train_idx = int(0.8 * n)
        
        # Initialize model
        self.model = CategoricalEmbeddingModel(vocab_sizes, self.EMBEDDING_DIMS).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_mse = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            
            # Shuffle
            perm = torch.randperm(train_idx)
            
            train_loss = 0
            n_batches = 0
            
            for i in range(0, train_idx, batch_size):
                idx = perm[i:i+batch_size]
                
                src_batch = {col: src_tensors[col][idx].to(self.device) for col in self.CATEGORICAL_COLS}
                dst_batch = {col: dst_tensors[col][idx].to(self.device) for col in self.CATEGORICAL_COLS}
                y_batch = y_tensor[idx].to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(src_batch, dst_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                src_val = {col: src_tensors[col][train_idx:].to(self.device) for col in self.CATEGORICAL_COLS}
                dst_val = {col: dst_tensors[col][train_idx:].to(self.device) for col in self.CATEGORICAL_COLS}
                val_pred = self.model(src_val, dst_val).cpu().numpy()
                val_mse = mean_squared_error(scores[train_idx:], val_pred)
            
            scheduler.step()
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val MSE: {val_mse:.6f}")
        
        print(f"\nBest Val MSE: {best_val_mse:.6f}")
        return self
    
    def encode(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Encode categorical columns using learned embeddings.
        
        Returns dict of {Profile_ID: {category: embedding_vector}}
        """
        self.model.eval()
        
        encoded_data = {}
        for _, row in df.iterrows():
            profile_id = row['Profile_ID']
            encoded_data[profile_id] = {}
            
            for col in self.CATEGORICAL_COLS:
                value = row[col] if not pd.isna(row[col]) else 'UNKNOWN'
                
                if value in self.label_encoders[col].classes_:
                    idx = self.label_encoders[col].transform([value])[0]
                else:
                    idx = 0  # Unknown
                
                with torch.no_grad():
                    idx_tensor = torch.LongTensor([idx]).to(self.device)
                    emb = self.model.embeddings[col](idx_tensor).cpu().numpy()[0]
                    encoded_data[profile_id][col] = emb
        
        return encoded_data
    
    def compute_similarity_features(self, src_id: int, dst_id: int, 
                                      profiles: Dict[int, Dict]) -> Dict[str, float]:
        """
        Compute similarity features for a pair using learned embeddings.
        """
        features = {}
        
        for col in self.CATEGORICAL_COLS:
            src_emb = profiles[src_id].get(col, np.zeros(self.EMBEDDING_DIMS[col]))
            dst_emb = profiles[dst_id].get(col, np.zeros(self.EMBEDDING_DIMS[col]))
            
            # Cosine similarity
            norm_src = np.linalg.norm(src_emb)
            norm_dst = np.linalg.norm(dst_emb)
            
            if norm_src > 0 and norm_dst > 0:
                cos_sim = np.dot(src_emb, dst_emb) / (norm_src * norm_dst)
            else:
                cos_sim = 0.0
            
            features[f'learned_{col.lower()}_sim'] = cos_sim
        
        return features
    
    def save(self, path: str = None):
        """Save encoder to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'learned_embeddings.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'model_state': self.model.state_dict() if self.model else None,
                'vocab_sizes': {col: len(le.classes_) for col, le in self.label_encoders.items()},
                'embedding_dims': self.EMBEDDING_DIMS
            }, f)
        print(f"Learned embeddings saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load encoder from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.label_encoders = data['label_encoders']
        
        if data['model_state']:
            instance.model = CategoricalEmbeddingModel(
                data['vocab_sizes'], 
                data['embedding_dims']
            ).to(instance.device)
            instance.model.load_state_dict(data['model_state'])
        
        print(f"Learned embeddings loaded from {path}")
        return instance


if __name__ == '__main__':
    from feature_engineering import load_data
    
    train_df, test_df, target_df = load_data()
    
    # Train learned embeddings
    encoder = LearnedCategoricalEncoder()
    encoder.fit(train_df, target_df, epochs=50)
    encoder.save()
    
    # Encode profiles
    profiles = encoder.encode(train_df)
    
    # Test similarity features
    sample_pairs = [(5001, 5002), (5001, 5003)]
    for src, dst in sample_pairs:
        features = encoder.compute_similarity_features(src, dst, profiles)
        print(f"\nPair ({src}, {dst}):")
        for name, val in features.items():
            print(f"  {name}: {val:.4f}")
