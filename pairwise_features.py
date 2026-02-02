"""
Pairwise Feature Construction Module for Compatibility Prediction.

This module constructs pairwise features from participant pairs:
- Structured features (age diff, seniority diff, matches, etc.)
- Business interest overlap (Jaccard, count)
- Semantic similarity features (3 directional cosine scores)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from attention_features import cross_attention_alignment, mutual_attention_score


def compute_jaccard(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two multi-hot vectors.
    
    Args:
        vec1: First multi-hot vector
        vec2: Second multi-hot vector
    
    Returns:
        Jaccard similarity in range [0, 1]
    """
    intersection = np.sum(np.minimum(vec1, vec2))
    union = np.sum(np.maximum(vec1, vec2))
    
    if union == 0:
        return 0.0
    return intersection / union


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
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def set_jaccard(s1: frozenset, s2: frozenset) -> float:
    """Compute Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 0.0
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def set_dice(s1: frozenset, s2: frozenset) -> float:
    """Compute Dice coefficient between two sets."""
    if not s1 and not s2:
        return 0.0
    total = len(s1) + len(s2)
    return 2 * len(s1 & s2) / total if total > 0 else 0.0


def set_overlap(s1: frozenset, s2: frozenset) -> float:
    """Compute overlap coefficient between two sets."""
    if not s1 or not s2:
        return 0.0
    min_size = min(len(s1), len(s2))
    return len(s1 & s2) / min_size if min_size > 0 else 0.0


class PairwiseFeatureBuilder:
    """Builds pairwise feature vectors from participant pairs."""
    
    # Feature names in order
    FEATURE_NAMES = [
        # Base features (16)
        'age_diff_abs',           # Absolute age difference
        'seniority_diff',         # Signed seniority difference (src - dst)
        'industry_match',         # Binary: same industry
        'location_match',         # Binary: same location
        'company_size_ratio',     # Log ratio of company sizes
        'gender_match',           # Binary: same gender
        'role_match',             # Binary: same role
        'company_match',          # Binary: same company
        'interests_jaccard',      # Jaccard similarity of interests
        'interests_overlap',      # Count of shared interests
        'interests_src_count',    # Number of interests for source
        'interests_dst_count',    # Number of interests for destination
        'cos_obj_obj',            # Cosine: objectives(src) vs objectives(dst)
        'cos_obj_con',            # Cosine: objectives(src) vs constraints(dst) [conflict signal]
        'cos_con_obj',            # Cosine: constraints(src) vs objectives(dst)
        'cos_con_con',            # Cosine: constraints(src) vs constraints(dst)
        
        # Feature crosses (8) - Phase 1 improvements
        'industry_x_interests',   # industry_match * interests_jaccard
        'location_x_role',        # location_match * role_match
        'semantic_x_interests',   # cos_obj_obj * interests_overlap
        'seniority_x_conflict',   # seniority_diff * cos_obj_con (asymmetric)
        'age_x_industry',         # normalized age_diff * industry_match
        'loc_ind_role_triple',    # location * industry * role (triple interaction)
        'semantic_alignment',     # (cos_obj_obj + cos_con_con) / 2
        'conflict_asymmetry',     # cos_obj_con - cos_con_obj (directional conflict)
        
        # Attention features (6) - Phase 3 improvements
        'attn_obj_to_con',        # src objectives attending to dst constraints
        'attn_con_to_obj',        # src constraints attending to dst objectives
        'mutual_obj_attn',        # bidirectional objectives attention
        'mutual_con_attn',        # bidirectional constraints attention
        'attn_asymmetry',         # absolute difference in directional attention
        'attn_combined',          # average of directional attention scores
        
        # Learned categorical embeddings (4) - New improvements
        'learned_industry_sim',   # Learned Industry embedding similarity
        'learned_role_sim',       # Learned Role embedding similarity
        'learned_location_sim',   # Learned Location embedding similarity
        'learned_company_sim',    # Learned Company embedding similarity
        
        # Contextual Semantic Features (10) - Phase 5 improvements
        'cos_obj_obj_x_sen_gap',       # Semantic align x Seniority gap
        'cos_obj_obj_x_role_asym',     # Semantic align x Role asymmetry
        'cos_obj_obj_x_industry',      # Semantic align x Industry match
        'cos_obj_con_x_sen_gap',       # Conflict x Seniority gap
        'cos_obj_con_x_role_asym',     # Conflict x Role asymmetry
        'cos_con_obj_x_sen_gap',       # Rev Conflict x Seniority gap
        'cos_con_obj_x_industry',      # Rev Conflict x Industry match
        'interests_jaccard_x_sen_gap', # Interests x Seniority gap
        'interests_overlap_x_industry',# Interests x Industry match
        'interests_jaccard_x_role_asym',# Interests x Role asymmetry
        
        # ============ RAW TEXT SET FEATURES (Phase 6 - Reference Notebook) ============
        # These are the KEY features from the reference notebook that achieved 0.00037 MSE
        
        # Jaccard on raw text sets (5)
        'j_all',                  # Jaccard(ALL, ALL) - union of BI+BO+CO
        'j_bi',                   # Jaccard(BI, BI) - Business Interests
        'j_bo',                   # Jaccard(BO, BO) - Business Objectives
        'j_co',                   # Jaccard(CO, CO) - Constraints
        'j_bi_bo',                # Jaccard(BI+BO, BI+BO)
        
        # Dice coefficient (3)
        'dice_all',               # Dice(ALL, ALL)
        'dice_bi',                # Dice(BI, BI)
        'dice_bo',                # Dice(BO, BO)
        
        # Overlap coefficient (2)
        'overlap_all',            # Overlap(ALL, ALL)
        'overlap_bi',             # Overlap(BI, BI)
        
        # Set sizes (8)
        'bi_size_1', 'bi_size_2', # |BI| for src and dst
        'bo_size_1', 'bo_size_2', # |BO| for src and dst
        'all_size_1', 'all_size_2', # |ALL| for src and dst
        'all_inter', 'all_union', # |ALL ∩ ALL|, |ALL ∪ ALL|
        
        # Cross-category features (3) - CRITICAL
        'bi_bo_cross',            # |BI₁ ∩ BO₂| + |BO₁ ∩ BI₂|
        'bi_co_cross',            # |BI₁ ∩ CO₂| + |CO₁ ∩ BI₂|
        'bo_co_cross',            # |BO₁ ∩ CO₂| + |CO₁ ∩ BO₂|
        
        # Polynomial features (5)
        'j_all_sq',               # j_all²
        'j_bi_sq',                # j_bi²
        'j_all_j_bi',             # j_all × j_bi
        'j_all_j_bo',             # j_all × j_bo
        'j_bi_j_bo',              # j_bi × j_bo
    ]
    
    def __init__(self, participants: Dict[int, Dict], 
                 objectives_embeddings: np.ndarray,
                 objectives_id_map: Dict[int, int],
                 constraints_embeddings: np.ndarray,
                 constraints_id_map: Dict[int, int],
                 learned_embeddings: Dict[int, Dict] = None):
        """
        Initialize with participant features and embeddings.
        
        Args:
            participants: Dict mapping Profile_ID to feature dict
            objectives_embeddings: Array of objectives embeddings
            objectives_id_map: Mapping from Profile_ID to embedding index
            constraints_embeddings: Array of constraints embeddings
            constraints_id_map: Mapping from Profile_ID to embedding index
            learned_embeddings: Optional dict of learned categorical embeddings
        """
        self.participants = participants
        self.obj_emb = objectives_embeddings
        self.obj_map = objectives_id_map
        self.con_emb = constraints_embeddings
        self.con_map = constraints_id_map
        self.learned_emb = learned_embeddings or {}
    
    def get_objectives_embedding(self, profile_id: int) -> np.ndarray:
        """Get objectives embedding for a profile."""
        idx = self.obj_map.get(profile_id)
        if idx is None:
            return np.zeros(768, dtype=np.float32)
        return self.obj_emb[idx]
    
    def get_constraints_embedding(self, profile_id: int) -> np.ndarray:
        """Get constraints embedding for a profile."""
        idx = self.con_map.get(profile_id)
        if idx is None:
            return np.zeros(768, dtype=np.float32)
        return self.con_emb[idx]
    
    def build_pairwise_features(self, src_id: int, dst_id: int) -> np.ndarray:
        """
        Build pairwise feature vector for (src_id, dst_id) pair.
        
        Args:
            src_id: Source profile ID
            dst_id: Destination profile ID
        
        Returns:
            numpy array of pairwise features
        """
        src = self.participants.get(src_id)
        dst = self.participants.get(dst_id)
        
        if src is None or dst is None:
            raise ValueError(f"Missing participant: src={src_id}, dst={dst_id}")
        
        features = []
        
        # 1. Age difference (absolute)
        features.append(abs(src['age'] - dst['age']))
        
        # 2. Seniority difference (signed: src - dst)
        features.append(src['seniority_numeric'] - dst['seniority_numeric'])
        
        # 3. Industry match
        features.append(1.0 if src['industry'] == dst['industry'] else 0.0)
        
        # 4. Location match
        features.append(1.0 if src['location'] == dst['location'] else 0.0)
        
        # 5. Company size ratio (log scale)
        src_size = max(src['company_size'], 1)
        dst_size = max(dst['company_size'], 1)
        features.append(np.log(src_size / dst_size + 1e-6))
        
        # 6. Gender match
        features.append(1.0 if src['gender'] == dst['gender'] else 0.0)
        
        # 7. Role match
        features.append(1.0 if src['role'] == dst['role'] else 0.0)
        
        # 8. Company match
        features.append(1.0 if src['company_name'] == dst['company_name'] else 0.0)
        
        # 9. Interests Jaccard similarity
        src_interests = src['interests_multihot']
        dst_interests = dst['interests_multihot']
        features.append(compute_jaccard(src_interests, dst_interests))
        
        # 10. Interests overlap count
        features.append(np.sum(np.minimum(src_interests, dst_interests)))
        
        # 11. Source interests count
        features.append(np.sum(src_interests))
        
        # 12. Destination interests count
        features.append(np.sum(dst_interests))
        
        # Get semantic embeddings
        src_obj = self.get_objectives_embedding(src_id)
        dst_obj = self.get_objectives_embedding(dst_id)
        src_con = self.get_constraints_embedding(src_id)
        dst_con = self.get_constraints_embedding(dst_id)
        
        # 13. Cosine: objectives(src) vs objectives(dst)
        cos_obj_obj = cosine_similarity(src_obj, dst_obj)
        features.append(cos_obj_obj)
        
        # 14. Cosine: objectives(src) vs constraints(dst) [CONFLICT SIGNAL]
        cos_obj_con = cosine_similarity(src_obj, dst_con)
        features.append(cos_obj_con)
        
        # 15. Cosine: constraints(src) vs objectives(dst)
        cos_con_obj = cosine_similarity(src_con, dst_obj)
        features.append(cos_con_obj)
        
        # 16. Cosine: constraints(src) vs constraints(dst)
        cos_con_con = cosine_similarity(src_con, dst_con)
        features.append(cos_con_con)
        
        # ============ FEATURE CROSSES (Phase 1) ============
        # Cache base features for crosses
        age_diff = abs(src['age'] - dst['age'])
        seniority_diff = src['seniority_numeric'] - dst['seniority_numeric']
        industry_match = 1.0 if src['industry'] == dst['industry'] else 0.0
        location_match = 1.0 if src['location'] == dst['location'] else 0.0
        role_match = 1.0 if src['role'] == dst['role'] else 0.0
        interests_jaccard = compute_jaccard(src_interests, dst_interests)
        interests_overlap = np.sum(np.minimum(src_interests, dst_interests))
        
        # 17. Industry × Interests synergy
        features.append(industry_match * interests_jaccard)
        
        # 18. Location × Role (colocated with same role)
        features.append(location_match * role_match)
        
        # 19. Semantic × Interests (both semantic and explicit alignment)
        features.append(cos_obj_obj * interests_overlap)
        
        # 20. Seniority × Conflict (asymmetric seniority-based conflict)
        features.append(seniority_diff * cos_obj_con)
        
        # 21. Age × Industry (age diff matters within same industry)
        features.append((age_diff / 50.0) * industry_match)  # Normalized
        
        # 22. Triple interaction: location × industry × role
        features.append(location_match * industry_match * role_match)
        
        # 23. Semantic alignment score
        features.append((cos_obj_obj + cos_con_con) / 2.0)
        
        # 24. Conflict asymmetry (directional conflict difference)
        features.append(cos_obj_con - cos_con_obj)
        
        # ============ ATTENTION FEATURES (Phase 3) ============
        # 25. Source objectives attending to destination constraints
        attn_obj_to_con = cross_attention_alignment(src_obj, dst_con)
        features.append(attn_obj_to_con)
        
        # 26. Source constraints attending to destination objectives
        attn_con_to_obj = cross_attention_alignment(src_con, dst_obj)
        features.append(attn_con_to_obj)
        
        # 27. Mutual attention between objectives
        features.append(mutual_attention_score(src_obj, dst_obj))
        
        # 28. Mutual attention between constraints
        features.append(mutual_attention_score(src_con, dst_con))
        
        # 29. Attention asymmetry
        features.append(abs(attn_obj_to_con - attn_con_to_obj))
        
        # 30. Combined directional attention
        features.append((attn_obj_to_con + attn_con_to_obj) / 2.0)
        
        # ============ LEARNED CATEGORICAL EMBEDDINGS ============
        # These use learned embeddings trained end-to-end on compatibility data
        learned_cats = ['Industry', 'Role', 'Location_City', 'Company_Name']
        
        for cat in learned_cats:
            src_emb = self.learned_emb.get(src_id, {}).get(cat)
            dst_emb = self.learned_emb.get(dst_id, {}).get(cat)
            
            if src_emb is not None and dst_emb is not None:
                # Cosine similarity of learned embeddings
                norm_src = np.linalg.norm(src_emb)
                norm_dst = np.linalg.norm(dst_emb)
                if norm_src > 0 and norm_dst > 0:
                    sim = np.dot(src_emb, dst_emb) / (norm_src * norm_dst)
                else:
                    sim = 0.0
            else:
                sim = 0.0
            features.append(sim)

        # ============ CONTEXTUAL SEMANTIC FEATURES (Phase 5) ============
        # Modulate semantic similarity by professional context
        
        # Define context variables
        seniority_gap = abs(seniority_diff)
        role_asym = 1.0 - role_match
        
        # 35. Semantic alignment x Seniority gap (Does alignment matter more/less if seniority differs?)
        features.append(cos_obj_obj * seniority_gap)
        
        # 36. Semantic alignment x Role asymmetry (Alignment across different roles)
        features.append(cos_obj_obj * role_asym)
        
        # 37. Semantic alignment x Industry (Alignment within same industry)
        features.append(cos_obj_obj * industry_match)
        
        # 38. Conflict (Obj-Con) x Seniority gap
        features.append(cos_obj_con * seniority_gap)
        
        # 39. Conflict (Obj-Con) x Role asym
        features.append(cos_obj_con * role_asym)
        
        # 40. Reverse Conflict (Con-Obj) x Seniority gap
        features.append(cos_con_obj * seniority_gap)
        
        # 41. Reverse Conflict (Con-Obj) x Industry
        features.append(cos_con_obj * industry_match)
        
        # 42. Interests Jaccard x Seniority gap
        features.append(interests_jaccard * seniority_gap)
        
        # 43. Interests Overlap x Industry
        features.append(interests_overlap * industry_match)
        
        # 44. Interests Jaccard x Role asymmetry
        features.append(interests_jaccard * role_asym)
        
        # ============ RAW TEXT SET FEATURES (Phase 6) ============
        # These are the KEY features from the reference notebook
        
        # Get raw text sets from participants
        src_bi = src.get('BI', frozenset())
        dst_bi = dst.get('BI', frozenset())
        src_bo = src.get('BO', frozenset())
        dst_bo = dst.get('BO', frozenset())
        src_co = src.get('CO', frozenset())
        dst_co = dst.get('CO', frozenset())
        src_all = src.get('ALL', frozenset())
        dst_all = dst.get('ALL', frozenset())
        src_bi_bo = src.get('BI_BO', frozenset())
        dst_bi_bo = dst.get('BI_BO', frozenset())
        
        # Jaccard similarities (5)
        j_all = set_jaccard(src_all, dst_all)
        j_bi = set_jaccard(src_bi, dst_bi)
        j_bo = set_jaccard(src_bo, dst_bo)
        j_co = set_jaccard(src_co, dst_co)
        j_bi_bo = set_jaccard(src_bi_bo, dst_bi_bo)
        
        features.append(j_all)
        features.append(j_bi)
        features.append(j_bo)
        features.append(j_co)
        features.append(j_bi_bo)
        
        # Dice coefficients (3)
        features.append(set_dice(src_all, dst_all))
        features.append(set_dice(src_bi, dst_bi))
        features.append(set_dice(src_bo, dst_bo))
        
        # Overlap coefficients (2)
        features.append(set_overlap(src_all, dst_all))
        features.append(set_overlap(src_bi, dst_bi))
        
        # Set sizes (8)
        features.append(len(src_bi))
        features.append(len(dst_bi))
        features.append(len(src_bo))
        features.append(len(dst_bo))
        features.append(len(src_all))
        features.append(len(dst_all))
        features.append(len(src_all & dst_all))  # Intersection
        features.append(len(src_all | dst_all))  # Union
        
        # Cross-category features (3) - CRITICAL
        features.append(len(src_bi & dst_bo) + len(src_bo & dst_bi))  # BI-BO cross
        features.append(len(src_bi & dst_co) + len(src_co & dst_bi))  # BI-CO cross
        features.append(len(src_bo & dst_co) + len(src_co & dst_bo))  # BO-CO cross
        
        # Polynomial features (5)
        features.append(j_all ** 2)
        features.append(j_bi ** 2)
        features.append(j_all * j_bi)
        features.append(j_all * j_bo)
        features.append(j_bi * j_bo)
        
        return np.array(features, dtype=np.float32)
    
    def build_pairwise_dataset(self, pairs_df: pd.DataFrame, 
                                show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build pairwise feature matrix for a DataFrame of pairs.
        
        Args:
            pairs_df: DataFrame with 'src_user_id', 'dst_user_id', and optionally 'compatibility_score'
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (feature_matrix, labels) where labels is None if no 'compatibility_score' column
        """
        n_pairs = len(pairs_df)
        n_features = len(self.FEATURE_NAMES)
        
        X = np.zeros((n_pairs, n_features), dtype=np.float32)
        
        has_labels = 'compatibility_score' in pairs_df.columns
        y = np.zeros(n_pairs, dtype=np.float32) if has_labels else None
        
        iterator = pairs_df.iterrows()
        if show_progress:
            iterator = tqdm(iterator, total=n_pairs, desc="Building pairwise features")
        
        for i, (_, row) in enumerate(iterator):
            src_id = row['src_user_id']
            dst_id = row['dst_user_id']
            
            X[i] = self.build_pairwise_features(src_id, dst_id)
            
            if has_labels:
                y[i] = row['compatibility_score']
        
        return X, y
    
    def build_all_pairs(self, profile_ids: List[int], 
                        show_progress: bool = True) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Build pairwise features for all ordered pairs of profiles.
        
        Args:
            profile_ids: List of profile IDs
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (feature_matrix, list of (src_id, dst_id) pairs)
        """
        # Generate all ordered pairs
        pairs = [(src, dst) for src in profile_ids for dst in profile_ids]
        n_pairs = len(pairs)
        n_features = len(self.FEATURE_NAMES)
        
        print(f"Building features for {n_pairs} pairs...")
        
        X = np.zeros((n_pairs, n_features), dtype=np.float32)
        
        iterator = enumerate(pairs)
        if show_progress:
            iterator = tqdm(iterator, total=n_pairs, desc="Building pairwise features")
        
        for i, (src_id, dst_id) in iterator:
            X[i] = self.build_pairwise_features(src_id, dst_id)
        
        return X, pairs


if __name__ == '__main__':
    # Test pairwise feature construction
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    
    # Load and process data
    train_df, test_df, target_df = load_data()
    
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    train_participants = processor.process_dataframe(train_df)
    
    # Get embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    # Build pairwise feature builder
    builder = PairwiseFeatureBuilder(
        participants=train_participants,
        objectives_embeddings=embeddings['train_objectives'],
        objectives_id_map=embeddings['train_objectives_id_map'],
        constraints_embeddings=embeddings['train_constraints'],
        constraints_id_map=embeddings['train_constraints_id_map']
    )
    
    # Test on sample pairs
    sample_pairs = target_df.head(5)
    X, y = builder.build_pairwise_dataset(sample_pairs, show_progress=False)
    
    print("\nFeature matrix shape:", X.shape)
    print("Labels shape:", y.shape)
    print("\nFeature names:", builder.FEATURE_NAMES)
    print("\nSample features (first pair):")
    for name, val in zip(builder.FEATURE_NAMES, X[0]):
        print(f"  {name}: {val:.4f}")
    print(f"\nLabel: {y[0]:.4f}")
