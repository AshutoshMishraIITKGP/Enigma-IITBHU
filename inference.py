"""
Inference Module for Pairwise Compatibility Prediction.

This module handles generating predictions for test set:
- Generate all ordered pairs from test profiles
- Apply identical feature construction
- Predict and clip to [0, 1]
- Save submission CSV
"""

import numpy as np
import pandas as pd
import os
from typing import List, Tuple
from tqdm import tqdm


def generate_all_pairs(profile_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Generate all ordered pairs from profile IDs.
    
    Args:
        profile_ids: List of profile IDs
    
    Returns:
        List of (src_id, dst_id) tuples
    """
    pairs = [(src, dst) for src in profile_ids for dst in profile_ids]
    return pairs


def create_submission(predictions: np.ndarray, 
                      pairs: List[Tuple[int, int]],
                      output_path: str = 'submission.csv') -> pd.DataFrame:
    """
    Create submission CSV file.
    
    Args:
        predictions: Array of predicted compatibility scores
        pairs: List of (src_id, dst_id) tuples
        output_path: Path to save submission CSV
    
    Returns:
        Submission DataFrame
    """
    # Create ID column: "src_user_id_dst_user_id"
    ids = [f"{src}_{dst}" for src, dst in pairs]
    
    submission = pd.DataFrame({
        'ID': ids,
        'compatibility_score': predictions
    })
    
    # Ensure proper formatting
    submission['compatibility_score'] = submission['compatibility_score'].clip(0, 1)
    
    # Save to CSV
    submission.to_csv(output_path, index=False)
    
    print(f"\nSubmission saved to {output_path}")
    print(f"  Rows: {len(submission)}")
    print(f"  Score range: [{submission['compatibility_score'].min():.4f}, {submission['compatibility_score'].max():.4f}]")
    print(f"  Score mean: {submission['compatibility_score'].mean():.4f}")
    
    return submission


def run_inference(model, pairwise_builder, test_profile_ids: List[int],
                  output_path: str = 'submission.csv') -> pd.DataFrame:
    """
    Run full inference pipeline.
    
    Args:
        model: Trained CompatibilityModel
        pairwise_builder: PairwiseFeatureBuilder configured for test data
        test_profile_ids: List of test profile IDs
        output_path: Path to save submission
    
    Returns:
        Submission DataFrame
    """
    print(f"\nStarting inference for {len(test_profile_ids)} profiles...")
    
    # Generate all pairs
    pairs = generate_all_pairs(test_profile_ids)
    n_pairs = len(pairs)
    print(f"Total pairs to predict: {n_pairs}")
    
    # Build features and predict in batches for memory efficiency
    batch_size = 10000
    n_batches = (n_pairs + batch_size - 1) // batch_size
    
    all_predictions = []
    
    for batch_idx in tqdm(range(n_batches), desc="Predicting"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_pairs)
        batch_pairs = pairs[start_idx:end_idx]
        
        # Build features for batch
        X_batch = np.array([
            pairwise_builder.build_pairwise_features(src, dst)
            for src, dst in batch_pairs
        ], dtype=np.float32)
        
        # Predict
        predictions_batch = model.predict(X_batch, clip=True)
        all_predictions.extend(predictions_batch)
    
    predictions = np.array(all_predictions)
    
    # Create submission
    submission = create_submission(predictions, pairs, output_path)
    
    return submission


def validate_submission(submission_path: str, expected_rows: int = None) -> bool:
    """
    Validate submission format.
    
    Args:
        submission_path: Path to submission CSV
        expected_rows: Expected number of rows (optional)
    
    Returns:
        True if valid, False otherwise
    """
    print(f"\nValidating submission: {submission_path}")
    
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"  ERROR: Could not read file: {e}")
        return False
    
    # Check columns
    if df.columns.tolist() != ['ID', 'compatibility_score']:
        print(f"  ERROR: Invalid columns. Expected ['ID', 'compatibility_score'], got {df.columns.tolist()}")
        return False
    print("  [OK] Columns correct")
    
    # Check row count
    if expected_rows is not None and len(df) != expected_rows:
        print(f"  ERROR: Invalid row count. Expected {expected_rows}, got {len(df)}")
        return False
    print(f"  [OK] Row count: {len(df)}")
    
    # Check ID format
    sample_ids = df['ID'].head(3).tolist()
    valid_format = all('_' in str(id) for id in sample_ids)
    if not valid_format:
        print(f"  ERROR: Invalid ID format. Sample: {sample_ids}")
        return False
    print(f"  [OK] ID format valid (sample: {sample_ids[0]})")
    
    # Check score range
    min_score = df['compatibility_score'].min()
    max_score = df['compatibility_score'].max()
    if min_score < 0 or max_score > 1:
        print(f"  ERROR: Scores out of range [0,1]: [{min_score}, {max_score}]")
        return False
    print(f"  [OK] Score range valid: [{min_score:.4f}, {max_score:.4f}]")
    
    # Check for missing values
    if df.isnull().any().any():
        print("  ERROR: Submission contains missing values")
        return False
    print("  [OK] No missing values")
    
    print("\n[OK] Submission is valid!")
    return True


if __name__ == '__main__':
    # Test inference pipeline (requires trained model)
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    from pairwise_features import PairwiseFeatureBuilder
    from model_training import CompatibilityModel
    
    # Check if model exists
    model_path = 'models/compatibility_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        exit(1)
    
    # Load data
    train_df, test_df, _ = load_data()
    
    # Process features
    processor = FeatureProcessor.load('models/feature_processor.pkl')
    test_participants = processor.process_dataframe(test_df)
    
    # Load cached embeddings
    generator = EmbeddingGenerator()
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    # Build pairwise feature builder for test data
    builder = PairwiseFeatureBuilder(
        participants=test_participants,
        objectives_embeddings=embeddings['test_objectives'],
        objectives_id_map=embeddings['test_objectives_id_map'],
        constraints_embeddings=embeddings['test_constraints'],
        constraints_id_map=embeddings['test_constraints_id_map']
    )
    
    # Load model
    model = CompatibilityModel.load(model_path)
    
    # Run inference
    test_profile_ids = test_df['Profile_ID'].tolist()
    submission = run_inference(model, builder, test_profile_ids)
    
    # Validate
    validate_submission('submission.csv', expected_rows=len(test_profile_ids) ** 2)
