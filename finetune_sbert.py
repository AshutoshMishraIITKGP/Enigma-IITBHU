"""
SBERT Fine-Tuning Module for Pairwise Compatibility Prediction.

Fine-tunes the sentence-transformers model on compatibility pairs
using CosineSimilarityLoss so that compatible profiles have similar embeddings.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
from tqdm import tqdm


def prepare_training_pairs(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    text_column: str = 'Business_Objectives',
    sample_size: int = 50000
) -> List[Tuple[str, str, float]]:
    """
    Create training pairs from target.csv for contrastive learning.
    
    Args:
        train_df: DataFrame with Profile_ID and text columns
        target_df: DataFrame with src_user_id, dst_user_id, compatibility_score
        text_column: Which text column to use
        sample_size: Number of pairs to sample (training efficiency)
    
    Returns:
        List of (text_a, text_b, score) tuples
    """
    # Create id -> text mapping
    id_to_text = {}
    for _, row in train_df.iterrows():
        text = str(row[text_column]) if not pd.isna(row[text_column]) else ""
        id_to_text[row['Profile_ID']] = text
    
    # Sample pairs for efficiency
    if len(target_df) > sample_size:
        sampled_df = target_df.sample(sample_size, random_state=42)
    else:
        sampled_df = target_df
    
    pairs = []
    for _, row in sampled_df.iterrows():
        src_id = row['src_user_id']
        dst_id = row['dst_user_id']
        score = row['compatibility_score']
        
        text_a = id_to_text.get(src_id, "")
        text_b = id_to_text.get(dst_id, "")
        
        if text_a and text_b:  # Skip empty texts
            pairs.append((text_a, text_b, score))
    
    print(f"Created {len(pairs)} training pairs from {text_column}")
    return pairs


def finetune_sbert(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    model_name: str = 'all-mpnet-base-v2',
    output_path: str = 'models/finetuned_sbert',
    epochs: int = 1,
    batch_size: int = 64,
    warmup_steps: int = 50,
    sample_size: int = 10000
):
    """
    Fine-tune SBERT model on compatibility pairs.
    
    Args:
        train_df: DataFrame with profile texts
        target_df: DataFrame with compatibility scores
        model_name: Base model to fine-tune
        output_path: Where to save fine-tuned model
        epochs: Training epochs (default: 1 for speed)
        batch_size: Batch size (default: 64)
        warmup_steps: Warmup steps for scheduler
        sample_size: Number of pairs per text field (default: 10000 for speed)
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    
    print(f"\nFine-tuning SBERT model: {model_name}")
    print(f"Output path: {output_path}")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Prepare training pairs for objectives
    obj_pairs = prepare_training_pairs(
        train_df, target_df, 
        text_column='Business_Objectives',
        sample_size=sample_size
    )
    
    # Also add constraint pairs for completeness
    con_pairs = prepare_training_pairs(
        train_df, target_df,
        text_column='Constraints', 
        sample_size=sample_size
    )
    
    # Combine pairs
    all_pairs = obj_pairs + con_pairs
    np.random.shuffle(all_pairs)
    
    # Split into train/eval
    split_idx = int(0.9 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    eval_pairs = all_pairs[split_idx:]
    
    print(f"Train pairs: {len(train_pairs)}, Eval pairs: {len(eval_pairs)}")
    
    # Create InputExamples
    train_examples = [
        InputExample(texts=[p[0], p[1]], label=float(p[2]))
        for p in train_pairs
    ]
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Loss function - CosineSimilarityLoss for regression
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Evaluator
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=[p[0] for p in eval_pairs],
        sentences2=[p[1] for p in eval_pairs],
        scores=[p[2] for p in eval_pairs],
        name='compatibility_eval'
    )
    
    # Train
    print("\nStarting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        evaluation_steps=500
    )
    
    print(f"\nFine-tuned model saved to {output_path}")
    return model


def get_finetuned_model(output_path: str = 'models/finetuned_sbert'):
    """Load fine-tuned model if exists, else return None."""
    from sentence_transformers import SentenceTransformer
    
    if os.path.exists(output_path):
        print(f"Loading fine-tuned model from {output_path}")
        return SentenceTransformer(output_path)
    return None


if __name__ == '__main__':
    from feature_engineering import load_data
    
    train_df, test_df, target_df = load_data()
    
    # Fine-tune on training data
    model = finetune_sbert(
        train_df, target_df,
        epochs=3,
        batch_size=32
    )
    
    # Test embeddings
    test_texts = [
        "Looking for investment partners in AI technology",
        "Seeking Series A funding for machine learning startup"
    ]
    embeddings = model.encode(test_texts)
    print(f"\nTest embeddings shape: {embeddings.shape}")
    
    # Similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"Similarity between test texts: {sim:.4f}")
