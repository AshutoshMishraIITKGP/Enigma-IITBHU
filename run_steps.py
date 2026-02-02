"""
Step-by-Step Training Pipeline with Checkpointing.

Run each step separately. Progress is cached, so you can:
- Re-run after a crash without losing progress
- Skip completed steps automatically

Usage:
    python run_steps.py step1     # Fine-tune SBERT
    python run_steps.py step2     # Train learned embeddings
    python run_steps.py step3     # Generate embeddings
    python run_steps.py step4     # Train ensemble model
    python run_steps.py all       # Run all steps
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Checkpoint file to track completed steps
CHECKPOINT_FILE = 'models/checkpoint.txt'


def save_checkpoint(step: str):
    """Mark a step as completed."""
    os.makedirs('models', exist_ok=True)
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(f"{step}|{datetime.now().isoformat()}\n")
    print(f"[CHECKPOINT] Saved: {step}")


def is_step_completed(step: str) -> bool:
    """Check if a step is already completed."""
    if not os.path.exists(CHECKPOINT_FILE):
        return False
    with open(CHECKPOINT_FILE, 'r') as f:
        completed = [line.split('|')[0] for line in f.readlines()]
    return step in completed


def reset_checkpoints():
    """Clear all checkpoints to restart from scratch."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    print("[RESET] All checkpoints cleared")


def print_header(text):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")


# ============== STEP 1: Fine-tune SBERT ==============
def step1_finetune_sbert():
    """Fine-tune SBERT on compatibility pairs."""
    step_name = 'step1_finetune_sbert'
    
    if is_step_completed(step_name):
        print(f"[SKIP] {step_name} already completed")
        return
    
    if os.path.exists('models/finetuned_sbert'):
        print("[SKIP] Fine-tuned model already exists")
        save_checkpoint(step_name)
        return
    
    print_header("STEP 1: Fine-tuning SBERT")
    
    from feature_engineering import load_data
    from finetune_sbert import finetune_sbert
    
    train_df, test_df, target_df = load_data()
    
    # Fast fine-tuning: 10k pairs, 1 epoch, batch 64
    finetune_sbert(
        train_df, target_df,
        epochs=1,
        batch_size=64,
        sample_size=10000  # Small for speed
    )
    
    save_checkpoint(step_name)


# ============== STEP 2: Train Learned Embeddings ==============
def step2_learned_embeddings():
    """Train learned categorical embeddings."""
    step_name = 'step2_learned_embeddings'
    
    if is_step_completed(step_name):
        print(f"[SKIP] {step_name} already completed")
        return
    
    if os.path.exists('models/learned_embeddings.pkl'):
        print("[SKIP] Learned embeddings already exist")
        save_checkpoint(step_name)
        return
    
    print_header("STEP 2: Training Learned Categorical Embeddings")
    
    from feature_engineering import load_data
    from learned_embeddings import LearnedCategoricalEncoder
    
    train_df, test_df, target_df = load_data()
    
    encoder = LearnedCategoricalEncoder()
    encoder.fit(train_df, target_df, epochs=30)
    encoder.save()
    
    save_checkpoint(step_name)


# ============== STEP 3: Generate Embeddings ==============
def step3_generate_embeddings():
    """Generate embeddings using fine-tuned model."""
    step_name = 'step3_generate_embeddings'
    
    if is_step_completed(step_name):
        print(f"[SKIP] {step_name} already completed")
        return
    
    print_header("STEP 3: Generating Semantic Embeddings")
    
    from feature_engineering import load_data
    from embedding_generation import EmbeddingGenerator
    
    train_df, test_df, target_df = load_data()
    
    # Use fine-tuned model if available
    finetuned_path = 'models/finetuned_sbert'
    model_name = finetuned_path if os.path.exists(finetuned_path) else None
    
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_all_embeddings(
        train_df, test_df, 
        force_regenerate=True  # Regenerate with new model
    )
    
    print(f"Generated embeddings: {embeddings['train_objectives'].shape}")
    save_checkpoint(step_name)


# ============== STEP 4: Train Ensemble Model ==============
def step4_train_model(trials: int = 50):
    """Train XGBoost + LightGBM ensemble."""
    step_name = 'step4_train_model'
    
    print_header(f"STEP 4: Training Ensemble Model ({trials} trials)")
    
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    from pairwise_features import PairwiseFeatureBuilder
    from model_training import EnsembleModel, prepare_training_data
    
    train_df, test_df, target_df = load_data()
    
    # Feature processor
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    train_participants = processor.process_dataframe(train_df)
    processor.save('models/feature_processor.pkl')
    
    # Load embeddings
    finetuned_path = 'models/finetuned_sbert'
    model_name = finetuned_path if os.path.exists(finetuned_path) else None
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    # Load learned embeddings
    learned_embeddings = None
    if os.path.exists('models/learned_embeddings.pkl'):
        from learned_embeddings import LearnedCategoricalEncoder
        encoder = LearnedCategoricalEncoder.load('models/learned_embeddings.pkl')
        learned_embeddings = encoder.encode(train_df)
    
    # Build pairwise features
    builder = PairwiseFeatureBuilder(
        participants=train_participants,
        objectives_embeddings=embeddings['train_objectives'],
        objectives_id_map=embeddings['train_objectives_id_map'],
        constraints_embeddings=embeddings['train_constraints'],
        constraints_id_map=embeddings['train_constraints_id_map'],
        learned_embeddings=learned_embeddings
    )
    
    # Prepare training data
    X_train, X_val, y_train, y_val = prepare_training_data(target_df, builder)
    print(f"Feature count: {X_train.shape[1]}")
    
    # Train ensemble
    model = EnsembleModel()
    metrics = model.train(
        X_train, y_train, X_val, y_val,
        feature_names=builder.FEATURE_NAMES,
        n_trials=trials
    )
    model.save()
    
    # Feature importance
    importance = model.get_feature_importance()
    print("\nTop Features:")
    print(importance.head(10).to_string())
    importance.to_csv('models/feature_importance.csv', index=False)
    
    save_checkpoint(step_name)
    
    return metrics


# ============== STEP 5: Generate Predictions ==============
def step5_predict():
    """Generate submission file."""
    step_name = 'step5_predict'
    
    print_header("STEP 5: Generating Predictions")
    
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    from pairwise_features import PairwiseFeatureBuilder
    from model_training import EnsembleModel
    from inference import run_inference, validate_submission
    
    train_df, test_df, _ = load_data()
    
    # Load components
    processor = FeatureProcessor.load('models/feature_processor.pkl')
    test_participants = processor.process_dataframe(test_df)
    
    finetuned_path = 'models/finetuned_sbert'
    model_name = finetuned_path if os.path.exists(finetuned_path) else None
    generator = EmbeddingGenerator(model_name=model_name)
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    # Learned embeddings for test
    learned_embeddings = None
    if os.path.exists('models/learned_embeddings.pkl'):
        from learned_embeddings import LearnedCategoricalEncoder
        encoder = LearnedCategoricalEncoder.load('models/learned_embeddings.pkl')
        learned_embeddings = encoder.encode(test_df)
    
    builder = PairwiseFeatureBuilder(
        participants=test_participants,
        objectives_embeddings=embeddings['test_objectives'],
        objectives_id_map=embeddings['test_objectives_id_map'],
        constraints_embeddings=embeddings['test_constraints'],
        constraints_id_map=embeddings['test_constraints_id_map'],
        learned_embeddings=learned_embeddings
    )
    
    model = EnsembleModel.load('models/ensemble_model.pkl')
    
    test_ids = test_df['Profile_ID'].tolist()
    submission = run_inference(model, builder, test_ids)
    
    validate_submission('submission.csv', expected_rows=len(test_ids)**2)
    
    save_checkpoint(step_name)


def main():
    parser = argparse.ArgumentParser(description='Step-by-step training with checkpointing')
    parser.add_argument('step', choices=['step1', 'step2', 'step3', 'step4', 'step5', 'all', 'reset'],
                        help='Which step to run')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials for step4')
    
    args = parser.parse_args()
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.step == 'reset':
        reset_checkpoints()
    elif args.step == 'step1':
        step1_finetune_sbert()
    elif args.step == 'step2':
        step2_learned_embeddings()
    elif args.step == 'step3':
        step3_generate_embeddings()
    elif args.step == 'step4':
        step4_train_model(args.trials)
    elif args.step == 'step5':
        step5_predict()
    elif args.step == 'all':
        step1_finetune_sbert()
        step2_learned_embeddings()
        step3_generate_embeddings()
        step4_train_model(args.trials)
        step5_predict()
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
