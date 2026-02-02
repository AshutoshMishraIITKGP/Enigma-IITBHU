"""
Two-Tower Neural Network Model for Pairwise Compatibility Prediction.

Architecture:
- Separate encoder towers for source and destination profiles
- Combination layer with [src, dst, src*dst, |src-dst|]
- Head MLP with sigmoid output

This learns task-specific profile representations rather than 
relying solely on hand-crafted features.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class TwoTowerNetwork(nn.Module):
    """Two-Tower Neural Network for pairwise compatibility."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, embedding_dim: int = 64):
        """
        Initialize the two-tower network.
        
        Args:
            input_dim: Number of input features per profile
            hidden_dim: Hidden layer dimension
            embedding_dim: Final embedding dimension for each tower
        """
        super().__init__()
        
        # Shared encoder architecture (but separate weights)
        self.src_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        self.dst_tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Combination head: [src, dst, src*dst, |src-dst|] = 4 * embedding_dim
        combined_dim = 4 * embedding_dim
        
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x):
        """
        Forward pass. Input x contains concatenated src and dst features.
        We split them and process through separate towers.
        """
        # Split input: first half is src, second half is dst features
        half_dim = x.shape[1] // 2
        x_src = x[:, :half_dim]
        x_dst = x[:, half_dim:]
        
        # Encode through towers
        src_emb = self.src_tower(x_src)
        dst_emb = self.dst_tower(x_dst)
        
        # Combine: [src, dst, src*dst, |src-dst|]
        combined = torch.cat([
            src_emb,
            dst_emb,
            src_emb * dst_emb,
            torch.abs(src_emb - dst_emb)
        ], dim=1)
        
        # Predict compatibility
        output = self.head(combined)
        return output.squeeze()


class TwoTowerModel:
    """Wrapper for Two-Tower model training and inference."""
    
    MODEL_DIR = 'models'
    
    def __init__(self, model_dir: str = None, device: str = None):
        self.model_dir = model_dir or self.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = None
        self.feature_names = None
        self.train_metrics = {}
        self.input_dim = None
    
    def _prepare_data(self, X_train, y_train, X_val, y_val, batch_size=256):
        """Prepare DataLoaders for training."""
        # For two-tower, we need to duplicate features (src and dst get same features)
        # In practice, this should be separate src/dst feature extraction
        # For simplicity, we use the pairwise features directly
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, X_train, y_train, X_val, y_val,
              feature_names=None, epochs=100, batch_size=256, lr=0.001):
        """
        Train the two-tower model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names for importance
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.feature_names = feature_names
        self.input_dim = X_train.shape[1]
        
        print(f"\nTraining Two-Tower Neural Network")
        print(f"  Input dim: {self.input_dim}")
        print(f"  Train size: {len(X_train)}")
        print(f"  Val size: {len(X_val)}")
        print(f"  Epochs: {epochs}")
        
        # Initialize model
        # We treat half the features as "src" and half as "dst" for the two towers
        self.model = TwoTowerNetwork(
            input_dim=self.input_dim // 2,
            hidden_dim=128,
            embedding_dim=64
        ).to(self.device)
        
        # Ensure input can be split evenly
        if self.input_dim % 2 != 0:
            # Pad features
            pad_size = 1
            X_train = np.pad(X_train, ((0, 0), (0, pad_size)), mode='constant')
            X_val = np.pad(X_val, ((0, 0), (0, pad_size)), mode='constant')
            self.input_dim += pad_size
            self.model = TwoTowerNetwork(
                input_dim=self.input_dim // 2,
                hidden_dim=128,
                embedding_dim=64
            ).to(self.device)
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(X_train, y_train, X_val, y_val, batch_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        best_val_mse = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_x)
            
            train_loss /= len(X_train)
            
            # Validation
            self.model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    output = self.model(batch_x)
                    val_preds.extend(output.cpu().numpy())
            
            val_preds = np.array(val_preds)
            val_mse = mean_squared_error(y_val, val_preds)
            
            scheduler.step()
            
            # Early stopping
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                patience_counter = 0
                # Save best model state
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train MSE: {train_loss:.6f}, Val MSE: {val_mse:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        self.model.load_state_dict(best_state)
        
        # Final evaluation
        y_train_pred = self.predict(X_train, clip=False)
        y_val_pred = self.predict(X_val, clip=False)
        
        self.train_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        }
        
        print(f"\n{'='*50}")
        print("Two-Tower Training Complete!")
        print(f"{'='*50}")
        print(f"Train MSE:  {self.train_metrics['train_mse']:.6f}")
        print(f"Val MSE:    {self.train_metrics['val_mse']:.6f}")
        print(f"Val RMSE:   {self.train_metrics['val_rmse']:.6f}")
        print(f"Val MAE:    {self.train_metrics['val_mae']:.6f}")
        print(f"{'='*50}")
        
        return self.train_metrics
    
    def predict(self, X, clip=True):
        """Predict compatibility scores."""
        self.model.eval()
        
        # Pad if necessary
        if X.shape[1] != self.input_dim:
            X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])), mode='constant')
        
        predictions = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                output = self.model(batch)
                predictions.extend(output.cpu().numpy())
        
        predictions = np.array(predictions)
        
        if clip:
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def get_feature_importance(self):
        """Approximate feature importance using gradient-based analysis."""
        # For neural networks, feature importance is less straightforward
        # Return uniform importance as placeholder
        if self.feature_names:
            n_features = len(self.feature_names)
            importance = np.ones(n_features) / n_features
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return pd.DataFrame({'feature': [], 'importance': []})
    
    def save(self, path=None):
        """Save model to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'neural_model.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model_state': self.model.state_dict(),
                'input_dim': self.input_dim,
                'feature_names': self.feature_names,
                'train_metrics': self.train_metrics
            }, f)
        
        print(f"Neural model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.input_dim = data['input_dim']
        instance.feature_names = data['feature_names']
        instance.train_metrics = data.get('train_metrics', {})
        
        instance.model = TwoTowerNetwork(
            input_dim=instance.input_dim // 2,
            hidden_dim=128,
            embedding_dim=64
        ).to(instance.device)
        instance.model.load_state_dict(data['model_state'])
        
        print(f"Neural model loaded from {path}")
        return instance


if __name__ == '__main__':
    # Test two-tower model
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    from pairwise_features import PairwiseFeatureBuilder
    from model_training import prepare_training_data
    
    train_df, test_df, target_df = load_data()
    
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    train_participants = processor.process_dataframe(train_df)
    
    generator = EmbeddingGenerator()
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    builder = PairwiseFeatureBuilder(
        participants=train_participants,
        objectives_embeddings=embeddings['train_objectives'],
        objectives_id_map=embeddings['train_objectives_id_map'],
        constraints_embeddings=embeddings['train_constraints'],
        constraints_id_map=embeddings['train_constraints_id_map']
    )
    
    X_train, X_val, y_train, y_val = prepare_training_data(target_df, builder)
    
    model = TwoTowerModel()
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=50)
    model.save()
