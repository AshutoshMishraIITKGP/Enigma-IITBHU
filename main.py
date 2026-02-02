"""
Main Pipeline Orchestration for Pairwise Compatibility Prediction.

This is the entry point for running the complete pipeline:
- python main.py train                    : Train XGBoost only
- python main.py train --model ensemble   : Train XGBoost + LightGBM ensemble  
- python main.py train --finetune         : Fine-tune SBERT + learn embeddings first
- python main.py predict                  : Run inference
- python main.py all --model ensemble     : Train + predict

Features:
- Modular design with clear separation
- SBERT fine-tuning on compatibility pairs
- Learned categorical embeddings
- Optuna hyperparameter tuning (100 trials default)
"""

import argparse
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import load_data, FeatureProcessor
from embedding_generation import EmbeddingGenerator
from pairwise_features import PairwiseFeatureBuilder
from model_training import CompatibilityModel, EnsembleModel, prepare_training_data
from inference import run_inference, validate_submission


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")


def run_finetuning(train_df, target_df):
    """Run SBERT fine-tuning and learned embeddings training."""
    from finetune_sbert import finetune_sbert, get_finetuned_model
    from learned_embeddings import LearnedCategoricalEncoder
    
    # Check if already fine-tuned
    finetuned_model = get_finetuned_model()
    if finetuned_model is None:
        print_header("Fine-tuning SBERT on Compatibility Pairs")
        finetune_sbert(train_df, target_df, epochs=3)
    else:
        print("Using existing fine-tuned SBERT model")
    
    # Train learned embeddings
    encoder_path = 'models/learned_embeddings.pkl'
    if not os.path.exists(encoder_path):
        print_header("Training Learned Categorical Embeddings")
        encoder = LearnedCategoricalEncoder()
        encoder.fit(train_df, target_df, epochs=50)
        encoder.save()
    else:
        print("Using existing learned embeddings")
        encoder = LearnedCategoricalEncoder.load(encoder_path)
    
    return encoder


def run_training_pipeline(n_trials: int = 100, timeout: int = 900, model_type: str = 'ensemble',
                          finetune: bool = False):
    """
    Run the complete training pipeline.
    
    Args:
        n_trials: Number of Optuna trials for hyperparameter tuning
        timeout: Timeout for tuning in seconds
        model_type: 'xgb', 'ensemble', or 'neural'
        finetune: Whether to fine-tune SBERT and learn embeddings first
    """
    print_header(f"TRAINING PIPELINE (Model: {model_type}, Finetune: {finetune})")
    start_time = datetime.now()
    
    # Step 1: Load data
    print_header("Step 1: Loading Data")
    train_df, test_df, target_df = load_data()
    
    # Step 2: Optional fine-tuning
    learned_encoder = None
    if finetune:
        learned_encoder = run_finetuning(train_df, target_df)
    
    # Step 3: Process features
    print_header("Step 3: Feature Engineering")
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    train_participants = processor.process_dataframe(train_df)
    processor.save('models/feature_processor.pkl')
    
    # Step 4: Generate embeddings (use fine-tuned if available)
    print_header("Step 4: Generating Semantic Embeddings")
    
    # Check for fine-tuned model
    finetuned_path = 'models/finetuned_sbert'
    use_finetuned = os.path.exists(finetuned_path)
    
    generator = EmbeddingGenerator()
    if use_finetuned:
        print(f"Using fine-tuned model from {finetuned_path}")
        generator.model_name = finetuned_path  # Override model path
    
    embeddings = generator.generate_all_embeddings(train_df, test_df, force_regenerate=use_finetuned)
    
    # Step 5: Get learned embeddings if available
    learned_embeddings = None
    if learned_encoder:
        print_header("Step 5: Extracting Learned Categorical Embeddings")
        learned_embeddings = learned_encoder.encode(train_df)
    elif os.path.exists('models/learned_embeddings.pkl'):
        from learned_embeddings import LearnedCategoricalEncoder
        learned_encoder = LearnedCategoricalEncoder.load('models/learned_embeddings.pkl')
        learned_embeddings = learned_encoder.encode(train_df)
    
    # Step 6: Build pairwise features
    print_header("Step 6: Building Pairwise Features")
    builder = PairwiseFeatureBuilder(
        participants=train_participants,
        objectives_embeddings=embeddings['train_objectives'],
        objectives_id_map=embeddings['train_objectives_id_map'],
        constraints_embeddings=embeddings['train_constraints'],
        constraints_id_map=embeddings['train_constraints_id_map'],
        learned_embeddings=learned_embeddings
    )
    
    # Step 7: Prepare training data
    print_header("Step 7: Preparing Training Data")
    X_train, X_val, y_train, y_val = prepare_training_data(target_df, builder)
    
    print(f"Feature count: {X_train.shape[1]}")
    print(f"Feature names: {builder.FEATURE_NAMES}")
    
    # Step 8: Train model
    if model_type == 'ensemble':
        print_header(f"Step 8: Training Ensemble ({n_trials} trials per model)")
        model = EnsembleModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=builder.FEATURE_NAMES,
            n_trials=n_trials // 2
        )
        model.save()
    elif model_type == 'neural':
        print_header("Step 8: Training Two-Tower Neural Network")
        from two_tower_model import TwoTowerModel
        model = TwoTowerModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=builder.FEATURE_NAMES,
            epochs=100
        )
        model.save()
    else:
        print_header(f"Step 8: Training XGBoost with Optuna ({n_trials} trials)")
        model = CompatibilityModel()
        metrics = model.train(
            X_train, y_train, X_val, y_val,
            feature_names=builder.FEATURE_NAMES,
            tune=True,
            n_trials=n_trials,
            timeout=timeout
        )
        model.save()
    
    # Step 9: Feature importance
    print_header("Step 9: Feature Importance Analysis")
    importance_df = model.get_feature_importance()
    print(importance_df.head(15).to_string())
    importance_df.to_csv('models/feature_importance.csv', index=False)
    
    # Summary
    elapsed = datetime.now() - start_time
    print_header("TRAINING COMPLETE")
    print(f"Model type: {model_type}")
    print(f"Fine-tuned: {finetune}")
    print(f"Total time: {elapsed}")
    print(f"Validation RMSE: {metrics['val_rmse']:.6f}")
    print(f"Validation MAE: {metrics['val_mae']:.6f}")
    
    return model


def run_prediction_pipeline(model_type: str = 'ensemble'):
    """Run the prediction/inference pipeline."""
    print_header(f"PREDICTION PIPELINE (Model: {model_type})")
    start_time = datetime.now()
    
    # Step 1: Load data
    print_header("Step 1: Loading Data")
    train_df, test_df, _ = load_data()
    
    # Step 2: Load feature processor
    print_header("Step 2: Loading Feature Processor")
    processor = FeatureProcessor.load('models/feature_processor.pkl')
    test_participants = processor.process_dataframe(test_df)
    
    # Step 3: Load/generate embeddings
    print_header("Step 3: Loading Semantic Embeddings")
    finetuned_path = 'models/finetuned_sbert'
    
    generator = EmbeddingGenerator()
    if os.path.exists(finetuned_path):
        generator.model_name = finetuned_path
    
    embeddings = generator.generate_all_embeddings(train_df, test_df)
    
    # Step 4: Load learned embeddings if available
    learned_embeddings = None
    if os.path.exists('models/learned_embeddings.pkl'):
        from learned_embeddings import LearnedCategoricalEncoder
        encoder = LearnedCategoricalEncoder.load('models/learned_embeddings.pkl')
        learned_embeddings = encoder.encode(test_df)
    
    # Step 5: Build pairwise feature builder for test
    print_header("Step 5: Building Pairwise Feature Builder")
    builder = PairwiseFeatureBuilder(
        participants=test_participants,
        objectives_embeddings=embeddings['test_objectives'],
        objectives_id_map=embeddings['test_objectives_id_map'],
        constraints_embeddings=embeddings['test_constraints'],
        constraints_id_map=embeddings['test_constraints_id_map'],
        learned_embeddings=learned_embeddings
    )
    
    # Step 6: Load model
    print_header("Step 6: Loading Trained Model")
    if model_type == 'ensemble':
        model = EnsembleModel.load('models/ensemble_model.pkl')
    elif model_type == 'neural':
        from two_tower_model import TwoTowerModel
        model = TwoTowerModel.load('models/neural_model.pkl')
    else:
        model = CompatibilityModel.load('models/compatibility_model.pkl')
    
    # Step 7: Run inference
    print_header("Step 7: Running Inference")
    test_profile_ids = test_df['Profile_ID'].tolist()
    submission = run_inference(model, builder, test_profile_ids)
    
    # Step 8: Validate submission
    print_header("Step 8: Validating Submission")
    expected_rows = len(test_profile_ids) ** 2
    is_valid = validate_submission('submission.csv', expected_rows=expected_rows)
    
    # Summary
    elapsed = datetime.now() - start_time
    print_header("PREDICTION COMPLETE")
    print(f"Model type: {model_type}")
    print(f"Total time: {elapsed}")
    print(f"Submission rows: {len(submission)}")
    print(f"Validation passed: {is_valid}")


def main():
    parser = argparse.ArgumentParser(
        description='Pairwise Compatibility Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                           # Train XGBoost (100 trials)
  python main.py train --model ensemble          # Train XGBoost+LightGBM ensemble
  python main.py train --finetune                # Fine-tune SBERT + learn embeddings first
  python main.py train --model ensemble --finetune --trials 100
  python main.py predict --model ensemble        # Run inference with ensemble
  python main.py all --model ensemble --finetune # Full pipeline
        """
    )
    
    parser.add_argument('command', choices=['train', 'predict', 'all'],
                        help='Pipeline command to run')
    parser.add_argument('--model', type=str, default='ensemble',
                        choices=['xgb', 'ensemble', 'neural'],
                        help='Model type: xgb, ensemble, or neural (default: ensemble)')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune SBERT and learn categorical embeddings first')
    parser.add_argument('--trials', type=int, default=100,
                        help='Number of Optuna trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=900,
                        help='Optuna timeout in seconds (default: 900)')
    
    args = parser.parse_args()
    
    print_header("PAIRWISE COMPATIBILITY PREDICTION SYSTEM")
    print(f"Command: {args.command}")
    print(f"Model: {args.model}")
    print(f"Fine-tune: {args.finetune}")
    print(f"Trials: {args.trials}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.command == 'train':
        run_training_pipeline(n_trials=args.trials, timeout=args.timeout, 
                              model_type=args.model, finetune=args.finetune)
    elif args.command == 'predict':
        run_prediction_pipeline(model_type=args.model)
    elif args.command == 'all':
        run_training_pipeline(n_trials=args.trials, timeout=args.timeout,
                              model_type=args.model, finetune=args.finetune)
        run_prediction_pipeline(model_type=args.model)


if __name__ == '__main__':
    main()
