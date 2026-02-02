"""
Feature Engineering Module for Pairwise Compatibility Prediction.

This module handles all participant-level feature processing:
- Data loading from Excel files
- Categorical encoding with LabelEncoders
- Multi-hot encoding for Business_Interests
- Numerical feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
import pickle
import os

# Constants
CATEGORICAL_COLS = ['Gender', 'Role', 'Seniority_Level', 'Company_Name', 'Industry', 'Location_City']
NUMERICAL_COLS = ['Age', 'Company_Size_Employees']
TEXT_COLS = ['Business_Objectives', 'Constraints']
MULTI_LABEL_COL = 'Business_Interests'

# Seniority level ordering (for signed difference computation)
SENIORITY_ORDER = {'Junior': 0, 'Mid-Level': 1, 'Senior': 2, 'Executive': 3}


class FeatureProcessor:
    """Handles all feature processing for participant profiles."""
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.interest_vocabulary: List[str] = []
        self.interest_to_idx: Dict[str, int] = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> 'FeatureProcessor':
        """
        Fit all encoders and scalers on training data.
        Optionally include test data for vocabulary building (interests only).
        
        Args:
            train_df: Training DataFrame with participant profiles
            test_df: Optional test DataFrame (used only for interest vocabulary)
        
        Returns:
            self for method chaining
        """
        # Combine for interest vocabulary (to avoid unseen interests at inference)
        if test_df is not None:
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            combined_df = train_df
        
        # 1. Build label encoders for categorical columns
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            # Fit on combined data to handle all possible values
            le.fit(combined_df[col].fillna('Unknown').astype(str))
            self.label_encoders[col] = le
        
        # 2. Build interest vocabulary from all data
        self._build_interest_vocabulary(combined_df)
        
        # 3. Fit scaler on training data only
        numerical_data = train_df[NUMERICAL_COLS].fillna(0)
        self.scaler.fit(numerical_data.values)
        
        self.is_fitted = True
        return self
    
    def _build_interest_vocabulary(self, df: pd.DataFrame) -> None:
        """Build global vocabulary of all unique Business_Interests."""
        all_interests = set()
        for interests_str in df[MULTI_LABEL_COL].dropna():
            interests = [i.strip() for i in str(interests_str).split(';') if i.strip()]
            all_interests.update(interests)
        
        self.interest_vocabulary = sorted(list(all_interests))
        self.interest_to_idx = {interest: idx for idx, interest in enumerate(self.interest_vocabulary)}
        print(f"Built interest vocabulary with {len(self.interest_vocabulary)} unique interests")
    
    def encode_multi_hot(self, interests_str: str) -> np.ndarray:
        """
        Convert semicolon-separated interests string to multi-hot vector.
        
        Args:
            interests_str: e.g., "AI;Data Science;SaaS"
        
        Returns:
            Binary numpy array of shape (n_interests,)
        """
        vector = np.zeros(len(self.interest_vocabulary), dtype=np.float32)
        if pd.isna(interests_str) or not interests_str:
            return vector
        
        interests = [i.strip() for i in str(interests_str).split(';') if i.strip()]
        for interest in interests:
            if interest in self.interest_to_idx:
                vector[self.interest_to_idx[interest]] = 1.0
        
        return vector
    
    def encode_categorical(self, value: str, col_name: str) -> int:
        """Encode a single categorical value."""
        if col_name not in self.label_encoders:
            raise ValueError(f"Unknown column: {col_name}")
        
        le = self.label_encoders[col_name]
        value_str = str(value) if not pd.isna(value) else 'Unknown'
        
        # Handle unseen values gracefully
        if value_str not in le.classes_:
            return -1  # Unknown category
        return le.transform([value_str])[0]
    
    def get_seniority_numeric(self, seniority: str) -> int:
        """Convert seniority level to numeric for signed difference."""
        if pd.isna(seniority):
            return 1  # Default to Mid-Level
        return SENIORITY_ORDER.get(str(seniority), 1)
    
    def process_participant(self, row: pd.Series) -> Dict:
        """
        Process a single participant row into feature dict.
        
        Args:
            row: pandas Series representing one participant
        
        Returns:
            Dict with all processed features
        """
        features = {
            'profile_id': row['Profile_ID'],
            
            # Raw numerical (for pairwise computation)
            'age': row['Age'],
            'company_size': row['Company_Size_Employees'] if not pd.isna(row['Company_Size_Employees']) else 0,
            
            # Scaled numerical
            'age_scaled': self.scaler.transform([[row['Age'], row.get('Company_Size_Employees', 0) or 0]])[0][0],
            'company_size_scaled': self.scaler.transform([[row['Age'], row.get('Company_Size_Employees', 0) or 0]])[0][1],
            
            # Raw categorical (for matching)
            'gender': row['Gender'],
            'role': row['Role'],
            'industry': row['Industry'],
            'location': row['Location_City'],
            'company_name': row['Company_Name'],
            
            # Encoded categorical
            'gender_encoded': self.encode_categorical(row['Gender'], 'Gender'),
            'role_encoded': self.encode_categorical(row['Role'], 'Role'),
            'seniority_encoded': self.encode_categorical(row['Seniority_Level'], 'Seniority_Level'),
            'industry_encoded': self.encode_categorical(row['Industry'], 'Industry'),
            'location_encoded': self.encode_categorical(row['Location_City'], 'Location_City'),
            'company_encoded': self.encode_categorical(row['Company_Name'], 'Company_Name'),
            
            # Seniority numeric (for signed difference)
            'seniority_numeric': self.get_seniority_numeric(row['Seniority_Level']),
            
            # Multi-hot interests
            'interests_multihot': self.encode_multi_hot(row[MULTI_LABEL_COL]),
            
            # Raw text (for embedding later)
            'objectives_text': str(row['Business_Objectives']) if not pd.isna(row['Business_Objectives']) else '',
            'constraints_text': str(row['Constraints']) if not pd.isna(row['Constraints']) else '',
        }
        
        return features
    
    def process_dataframe(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """
        Process entire DataFrame into dict of participant features.
        
        Args:
            df: DataFrame with participant profiles
        
        Returns:
            Dict mapping Profile_ID -> feature dict
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureProcessor must be fitted before processing")
        
        participants = {}
        for _, row in df.iterrows():
            features = self.process_participant(row)
            participants[features['profile_id']] = features
        
        print(f"Processed {len(participants)} participants")
        return participants
    
    def save(self, path: str) -> None:
        """Save fitted processor to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'label_encoders': self.label_encoders,
                'interest_vocabulary': self.interest_vocabulary,
                'interest_to_idx': self.interest_to_idx,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
        print(f"Saved FeatureProcessor to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureProcessor':
        """Load fitted processor from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        processor = cls()
        processor.label_encoders = data['label_encoders']
        processor.interest_vocabulary = data['interest_vocabulary']
        processor.interest_to_idx = data['interest_to_idx']
        processor.scaler = data['scaler']
        processor.is_fitted = data['is_fitted']
        
        print(f"Loaded FeatureProcessor from {path}")
        return processor


def load_data(data_dir: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, test, and target data from files.
    
    Args:
        data_dir: Directory containing data files
    
    Returns:
        Tuple of (train_df, test_df, target_df)
    """
    train_path = os.path.join(data_dir, 'train.xlsx')
    test_path = os.path.join(data_dir, 'test.xlsx')
    target_path = os.path.join(data_dir, 'target.csv')
    
    print("Loading data files...")
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    target_df = pd.read_csv(target_path)
    
    print(f"  Train: {len(train_df)} profiles")
    print(f"  Test: {len(test_df)} profiles")
    print(f"  Target: {len(target_df)} pairs")
    
    return train_df, test_df, target_df


if __name__ == '__main__':
    # Test the feature processor
    train_df, test_df, target_df = load_data()
    
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    
    # Process all participants
    train_participants = processor.process_dataframe(train_df)
    test_participants = processor.process_dataframe(test_df)
    
    # Show sample
    sample_id = list(train_participants.keys())[0]
    sample = train_participants[sample_id]
    print(f"\nSample participant {sample_id}:")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, sum={v.sum()}")
        else:
            print(f"  {k}: {v}")
    
    # Save processor
    processor.save('models/feature_processor.pkl')
