"""
Attention Features Module for Pairwise Compatibility Prediction.

Implements asymmetric cross-attention between text embeddings:
- Objectives attending to Constraints (and vice versa)
- Captures directional compatibility signals
"""

import numpy as np
from typing import Dict, Tuple, Optional


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray, 
    key: np.ndarray, 
    value: np.ndarray,
    temperature: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.
    
    Args:
        query: Query vector (d,)
        key: Key vector (d,)
        value: Value vector (d,)
        temperature: Scaling temperature (higher = softer attention)
    
    Returns:
        Tuple of (attended_output, attention_score)
    """
    d_k = query.shape[-1]
    
    # Scaled dot product (scalar attention score)
    score = np.dot(query, key) / (np.sqrt(d_k) * temperature)
    
    # For single key-value, attention weight is 1 after softmax
    # But we return the raw alignment score
    attn_weight = 1.0 / (1.0 + np.exp(-score))  # Sigmoid for bounded score
    
    # Attended value (weighted by attention)
    attended = attn_weight * value
    
    return attended, score


def cross_attention_alignment(
    query_emb: np.ndarray,
    key_emb: np.ndarray,
    value_emb: np.ndarray = None
) -> float:
    """
    Compute cross-attention alignment score.
    
    query attends to key/value to produce aligned representation,
    then we measure how well the attended output aligns with query.
    
    Args:
        query_emb: Query embedding (768,)
        key_emb: Key embedding (768,)
        value_emb: Value embedding (768,), defaults to key_emb
    
    Returns:
        Alignment score in range [-1, 1]
    """
    if value_emb is None:
        value_emb = key_emb
    
    # Compute attention
    attended, raw_score = scaled_dot_product_attention(query_emb, key_emb, value_emb)
    
    # Measure alignment between query and attended output
    query_norm = np.linalg.norm(query_emb)
    attended_norm = np.linalg.norm(attended)
    
    if query_norm == 0 or attended_norm == 0:
        return 0.0
    
    alignment = np.dot(query_emb, attended) / (query_norm * attended_norm)
    
    return float(alignment)


def mutual_attention_score(
    emb_a: np.ndarray,
    emb_b: np.ndarray
) -> float:
    """
    Bidirectional attention agreement score.
    
    Measures how much A attending to B agrees with B attending to A.
    High score = symmetric compatibility
    Low score = asymmetric relationship
    
    Args:
        emb_a: First embedding (768,)
        emb_b: Second embedding (768,)
    
    Returns:
        Mutual attention score
    """
    a_to_b = cross_attention_alignment(emb_a, emb_b)
    b_to_a = cross_attention_alignment(emb_b, emb_a)
    
    # Geometric mean of alignment scores (penalizes asymmetry)
    if a_to_b <= 0 or b_to_a <= 0:
        return 0.0
    
    return float(np.sqrt(a_to_b * b_to_a))


class AttentionFeatureGenerator:
    """Generates attention-based pairwise features."""
    
    def __init__(
        self,
        objectives_embeddings: np.ndarray,
        objectives_id_map: Dict[int, int],
        constraints_embeddings: np.ndarray,
        constraints_id_map: Dict[int, int]
    ):
        """
        Initialize with embeddings and ID mappings.
        
        Args:
            objectives_embeddings: Array of objectives embeddings
            objectives_id_map: Profile_ID -> embedding row index
            constraints_embeddings: Array of constraints embeddings  
            constraints_id_map: Profile_ID -> embedding row index
        """
        self.obj_emb = objectives_embeddings
        self.obj_map = objectives_id_map
        self.con_emb = constraints_embeddings
        self.con_map = constraints_id_map
    
    def get_objectives(self, profile_id: int) -> np.ndarray:
        """Get objectives embedding for a profile."""
        idx = self.obj_map.get(profile_id)
        if idx is None:
            return np.zeros(768, dtype=np.float32)
        return self.obj_emb[idx]
    
    def get_constraints(self, profile_id: int) -> np.ndarray:
        """Get constraints embedding for a profile."""
        idx = self.con_map.get(profile_id)
        if idx is None:
            return np.zeros(768, dtype=np.float32)
        return self.con_emb[idx]
    
    def compute_attention_features(self, src_id: int, dst_id: int) -> Dict[str, float]:
        """
        Compute all attention-based features for a pair.
        
        Args:
            src_id: Source profile ID
            dst_id: Destination profile ID
        
        Returns:
            Dict of feature name -> value
        """
        src_obj = self.get_objectives(src_id)
        dst_obj = self.get_objectives(dst_id)
        src_con = self.get_constraints(src_id)
        dst_con = self.get_constraints(dst_id)
        
        features = {}
        
        # 1. Source objectives attending to destination constraints
        # "What does dst require that src can provide?"
        features['attn_obj_to_con'] = cross_attention_alignment(src_obj, dst_con)
        
        # 2. Source constraints attending to destination objectives
        # "What does dst offer that satisfies src's constraints?"
        features['attn_con_to_obj'] = cross_attention_alignment(src_con, dst_obj)
        
        # 3. Mutual attention between objectives
        # "Do their goals align bidirectionally?"
        features['mutual_obj_attn'] = mutual_attention_score(src_obj, dst_obj)
        
        # 4. Mutual attention between constraints
        # "Do their constraints align?"
        features['mutual_con_attn'] = mutual_attention_score(src_con, dst_con)
        
        # 5. Asymmetry score: how different is directional attention?
        features['attn_asymmetry'] = abs(
            features['attn_obj_to_con'] - features['attn_con_to_obj']
        )
        
        # 6. Combined directional score
        features['attn_combined'] = (
            features['attn_obj_to_con'] + features['attn_con_to_obj']
        ) / 2.0
        
        return features
    
    def get_feature_names(self) -> list:
        """Get list of attention feature names."""
        return [
            'attn_obj_to_con',
            'attn_con_to_obj', 
            'mutual_obj_attn',
            'mutual_con_attn',
            'attn_asymmetry',
            'attn_combined'
        ]


if __name__ == '__main__':
    # Test attention features
    from feature_engineering import load_data
    from embedding_generation import EmbeddingGenerator
    
    train_df, test_df, target_df = load_data()
    
    generator = EmbeddingGenerator()
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    attn_gen = AttentionFeatureGenerator(
        objectives_embeddings=embeddings['train_objectives'],
        objectives_id_map=embeddings['train_objectives_id_map'],
        constraints_embeddings=embeddings['train_constraints'],
        constraints_id_map=embeddings['train_constraints_id_map']
    )
    
    # Test on sample pairs
    sample_pairs = [(5001, 5002), (5001, 5003), (5001, 5001)]
    
    print("Attention Features Test:")
    print("-" * 50)
    for src, dst in sample_pairs:
        features = attn_gen.compute_attention_features(src, dst)
        print(f"\nPair ({src}, {dst}):")
        for name, value in features.items():
            print(f"  {name}: {value:.4f}")
