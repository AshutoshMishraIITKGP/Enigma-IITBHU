"""
Model Training Module for Pairwise Compatibility Prediction.

This module handles model training with Optuna hyperparameter optimization:
- XGBoost regressor for compatibility score prediction
- Optuna for hyperparameter tuning
- Early stopping with validation set
- Model persistence
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

# Suppress Optuna logs during tuning
optuna.logging.set_verbosity(optuna.logging.WARNING)


class CompatibilityModel:
    """XGBoost-based compatibility prediction model with Optuna tuning."""
    
    MODEL_DIR = 'models'
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.best_params = None
        self.feature_names = None
        self.train_metrics = {}
    
    def _objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Optuna objective function for hyperparameter tuning."""
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 0,
            
            # Hyperparameters to tune
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        }
        
        # Train with early stopping
        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=50,
            random_state=42
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        
        return mse
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              n_trials: int = 50, timeout: int = 600) -> Dict:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
        
        Returns:
            Best hyperparameters dict
        """
        print(f"\nStarting Optuna hyperparameter tuning ({n_trials} trials, {timeout}s timeout)...")
        
        study = optuna.create_study(direction='minimize', study_name='compatibility_model')
        
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\nBest MSE: {study.best_value:.6f}")
        print(f"Best params: {self.best_params}")
        
        # Save study results
        study_path = os.path.join(self.model_dir, 'optuna_study.json')
        with open(study_path, 'w') as f:
            json.dump({
                'best_value': study.best_value,
                'best_params': self.best_params,
                'n_trials': len(study.trials),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return self.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              feature_names: list = None,
              tune: bool = True, n_trials: int = 50, timeout: int = 600) -> Dict:
        """
        Train the model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Optional list of feature names
            tune: Whether to run Optuna tuning
            n_trials: Number of Optuna trials
            timeout: Timeout for tuning in seconds
        
        Returns:
            Dict with training metrics
        """
        self.feature_names = feature_names
        
        if tune:
            # Run hyperparameter tuning
            self.tune_hyperparameters(X_train, y_train, X_val, y_val, n_trials, timeout)
        else:
            # Use default parameters
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.01,
                'reg_lambda': 1.0,
                'gamma': 0.01,
            }
        
        print("\nTraining final model with best parameters...")
        
        # Train final model with best params
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            booster='gbtree',
            verbosity=1,
            early_stopping_rounds=100,
            random_state=42,
            **self.best_params
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        self.train_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'best_iteration': self.model.best_iteration,
        }
        
        print(f"\n{'='*50}")
        print("Training Complete!")
        print(f"{'='*50}")
        print(f"Train MSE:  {self.train_metrics['train_mse']:.6f}")
        print(f"Train RMSE: {self.train_metrics['train_rmse']:.6f}")
        print(f"Train MAE:  {self.train_metrics['train_mae']:.6f}")
        print(f"Val MSE:    {self.train_metrics['val_mse']:.6f}")
        print(f"Val RMSE:   {self.train_metrics['val_rmse']:.6f}")
        print(f"Val MAE:    {self.train_metrics['val_mae']:.6f}")
        print(f"Best iter:  {self.train_metrics['best_iteration']}")
        print(f"{'='*50}")
        
        return self.train_metrics
    
    def predict(self, X: np.ndarray, clip: bool = True) -> np.ndarray:
        """
        Predict compatibility scores.
        
        Args:
            X: Feature matrix
            clip: Whether to clip predictions to [0, 1]
        
        Returns:
            Predicted compatibility scores
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        
        predictions = self.model.predict(X)
        
        if clip:
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")
        
        importance = self.model.feature_importances_
        
        if self.feature_names:
            names = self.feature_names
        else:
            names = [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str = None) -> None:
        """Save model and metadata to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'compatibility_model.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'train_metrics': self.train_metrics
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'CompatibilityModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.model = data['model']
        instance.best_params = data['best_params']
        instance.feature_names = data['feature_names']
        instance.train_metrics = data.get('train_metrics', {})
        
        print(f"Model loaded from {path}")
        return instance


class EnsembleModel:
    """Ensemble of XGBoost + LightGBM for improved predictions."""
    
    MODEL_DIR = 'models'
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.xgb_model = None
        self.lgb_model = None
        self.weights = [0.5, 0.5]  # Will be optimized
        self.feature_names = None
        self.train_metrics = {}
    
    def _tune_xgb(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for XGBoost."""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 0,
            'max_depth': trial.suggest_int('xgb_max_depth', 4, 8),
            'learning_rate': trial.suggest_float('xgb_lr', 0.02, 0.2, log=True),
            'n_estimators': trial.suggest_int('xgb_n_est', 200, 800),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.7, 0.95),
            'reg_lambda': trial.suggest_float('xgb_lambda', 0.1, 10.0, log=True),
        }
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model, mean_squared_error(y_val, model.predict(X_val))
    
    def _tune_lgb(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective for LightGBM."""
        import lightgbm as lgb
        
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': trial.suggest_int('lgb_max_depth', 4, 8),
            'learning_rate': trial.suggest_float('lgb_lr', 0.02, 0.2, log=True),
            'n_estimators': trial.suggest_int('lgb_n_est', 200, 800),
            'subsample': trial.suggest_float('lgb_subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('lgb_colsample', 0.7, 0.95),
            'reg_lambda': trial.suggest_float('lgb_lambda', 0.1, 10.0, log=True),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
        }
        
        model = lgb.LGBMRegressor(**params, random_state=42)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return model, mean_squared_error(y_val, model.predict(X_val))
    
    def _optimize_weights(self, X_val, y_val):
        """Find optimal ensemble weights using grid search."""
        best_mse = float('inf')
        best_weights = [0.5, 0.5]
        
        xgb_pred = self.xgb_model.predict(X_val)
        lgb_pred = self.lgb_model.predict(X_val)
        
        for w in np.arange(0.1, 0.95, 0.05):
            ensemble_pred = w * xgb_pred + (1 - w) * lgb_pred
            mse = mean_squared_error(y_val, ensemble_pred)
            if mse < best_mse:
                best_mse = mse
                best_weights = [w, 1 - w]
        
        self.weights = best_weights
        print(f"Optimal weights: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}")
        return best_weights
    
    def train(self, X_train, y_train, X_val, y_val, 
              feature_names=None, n_trials=25, timeout=300):
        """Train ensemble with Optuna tuning for both models."""
        import lightgbm as lgb
        
        self.feature_names = feature_names
        
        print("\n" + "="*50)
        print("Training XGBoost + LightGBM Ensemble")
        print("="*50)
        
        # Tune XGBoost
        print("\nTuning XGBoost...")
        xgb_study = optuna.create_study(direction='minimize')
        best_xgb_model = None
        best_xgb_mse = float('inf')
        
        for _ in range(n_trials):
            trial = xgb_study.ask()
            model, mse = self._tune_xgb(trial, X_train, y_train, X_val, y_val)
            xgb_study.tell(trial, mse)
            if mse < best_xgb_mse:
                best_xgb_mse = mse
                best_xgb_model = model
        
        self.xgb_model = best_xgb_model
        print(f"XGBoost best MSE: {best_xgb_mse:.6f}")
        
        # Tune LightGBM
        print("\nTuning LightGBM...")
        lgb_study = optuna.create_study(direction='minimize')
        best_lgb_model = None
        best_lgb_mse = float('inf')
        
        for _ in range(n_trials):
            trial = lgb_study.ask()
            model, mse = self._tune_lgb(trial, X_train, y_train, X_val, y_val)
            lgb_study.tell(trial, mse)
            if mse < best_lgb_mse:
                best_lgb_mse = mse
                best_lgb_model = model
        
        self.lgb_model = best_lgb_model
        print(f"LightGBM best MSE: {best_lgb_mse:.6f}")
        
        # Optimize weights
        print("\nOptimizing ensemble weights...")
        self._optimize_weights(X_val, y_val)
        
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
            'xgb_weight': self.weights[0],
            'lgb_weight': self.weights[1],
        }
        
        print(f"\n{'='*50}")
        print("Ensemble Training Complete!")
        print(f"{'='*50}")
        print(f"Train MSE:  {self.train_metrics['train_mse']:.6f}")
        print(f"Val MSE:    {self.train_metrics['val_mse']:.6f}")
        print(f"Val RMSE:   {self.train_metrics['val_rmse']:.6f}")
        print(f"Val MAE:    {self.train_metrics['val_mae']:.6f}")
        print(f"Weights:    XGB={self.weights[0]:.2f}, LGB={self.weights[1]:.2f}")
        print(f"{'='*50}")
        
        return self.train_metrics
    
    def predict(self, X, clip=True):
        """Weighted ensemble prediction."""
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)
        
        predictions = self.weights[0] * xgb_pred + self.weights[1] * lgb_pred
        
        if clip:
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def get_feature_importance(self):
        """Combined feature importance from both models."""
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        
        # Normalize and combine
        xgb_imp = xgb_imp / xgb_imp.sum()
        lgb_imp = lgb_imp / lgb_imp.sum()
        combined = self.weights[0] * xgb_imp + self.weights[1] * lgb_imp
        
        names = self.feature_names or [f'feature_{i}' for i in range(len(combined))]
        
        return pd.DataFrame({
            'feature': names,
            'importance': combined,
            'xgb_importance': xgb_imp,
            'lgb_importance': lgb_imp
        }).sort_values('importance', ascending=False)
    
    def save(self, path=None):
        """Save ensemble to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'ensemble_model.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'xgb_model': self.xgb_model,
                'lgb_model': self.lgb_model,
                'weights': self.weights,
                'feature_names': self.feature_names,
                'train_metrics': self.train_metrics
            }, f)
        
        print(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load ensemble from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.xgb_model = data['xgb_model']
        instance.lgb_model = data['lgb_model']
        instance.weights = data['weights']
        instance.feature_names = data['feature_names']
        instance.train_metrics = data.get('train_metrics', {})
        
        print(f"Ensemble loaded from {path}")
        return instance


class StackingEnsembleModel:
    """
    Multi-model stacking ensemble with Ridge meta-learner.
    Models: XGBoost, LightGBM, CatBoost, ExtraTrees, GradientBoosting
    """
    
    MODEL_DIR = 'models'
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}
        self.meta_model = None
        self.scaler = None
        self.feature_names = None
        self.train_metrics = {}
        self.model_mses = {}
    
    def train(self, X_train, y_train, X_val, y_val, 
              feature_names=None, n_trials=25, timeout=300):
        """Train stacking ensemble with multiple models."""
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        
        # Try importing CatBoost
        try:
            from catboost import CatBoostRegressor
            HAS_CATBOOST = True
        except ImportError:
            HAS_CATBOOST = False
            print("CatBoost not available, skipping...")
        
        self.feature_names = feature_names
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\n" + "="*60)
        print("Training Multi-Model Stacking Ensemble")
        print("="*60)
        
        # Combine train and val for OOF predictions
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        
        oof_predictions = {}
        
        # ============ MODEL 1: XGBoost (improved params) ============
        print("\n[1/5] Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            verbosity=0
        )
        oof_xgb = cross_val_predict(xgb_model, X_full, y_full, cv=kf)
        xgb_model.fit(X_full, y_full)
        xgb_mse = mean_squared_error(y_full, oof_xgb)
        print(f"    XGBoost OOF MSE: {xgb_mse:.10f}")
        oof_predictions['xgb'] = oof_xgb
        self.models['xgb'] = xgb_model
        self.model_mses['xgb'] = xgb_mse
        
        # ============ MODEL 2: LightGBM ============
        print("\n[2/5] Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            verbosity=-1
        )
        oof_lgb = cross_val_predict(lgb_model, X_full, y_full, cv=kf)
        lgb_model.fit(X_full, y_full)
        lgb_mse = mean_squared_error(y_full, oof_lgb)
        print(f"    LightGBM OOF MSE: {lgb_mse:.10f}")
        oof_predictions['lgb'] = oof_lgb
        self.models['lgb'] = lgb_model
        self.model_mses['lgb'] = lgb_mse
        
        # ============ MODEL 3: CatBoost ============
        if HAS_CATBOOST:
            print("\n[3/5] Training CatBoost...")
            cat_model = CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=0
            )
            oof_cat = cross_val_predict(cat_model, X_full, y_full, cv=kf)
            cat_model.fit(X_full, y_full)
            cat_mse = mean_squared_error(y_full, oof_cat)
            print(f"    CatBoost OOF MSE: {cat_mse:.10f}")
            oof_predictions['catboost'] = oof_cat
            self.models['catboost'] = cat_model
            self.model_mses['catboost'] = cat_mse
        else:
            print("\n[3/5] Skipping CatBoost (not installed)")
        
        # ============ MODEL 4: ExtraTrees (smaller for speed) ============
        print("\n[4/5] Training ExtraTrees...")
        et_model = ExtraTreesRegressor(
            n_estimators=100,  # Reduced from 300
            max_depth=8,       # Reduced from 10
            min_samples_split=10,  # Increased from 5
            random_state=42,
            n_jobs=-1
        )
        oof_et = cross_val_predict(et_model, X_full, y_full, cv=kf)
        et_model.fit(X_full, y_full)
        et_mse = mean_squared_error(y_full, oof_et)
        print(f"    ExtraTrees OOF MSE: {et_mse:.10f}")
        oof_predictions['et'] = oof_et
        self.models['et'] = et_model
        self.model_mses['et'] = et_mse
        
        # ============ MODEL 5: GradientBoosting ============
        print("\n[5/5] Training GradientBoosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        oof_gb = cross_val_predict(gb_model, X_full, y_full, cv=kf)
        gb_model.fit(X_full, y_full)
        gb_mse = mean_squared_error(y_full, oof_gb)
        print(f"    GradientBoosting OOF MSE: {gb_mse:.10f}")
        oof_predictions['gb'] = oof_gb
        self.models['gb'] = gb_model
        self.model_mses['gb'] = gb_mse
        
        # ============ META-LEARNER: Ridge on OOF predictions ============
        print("\n[META] Training Ridge meta-learner...")
        stack_features = np.column_stack([oof_predictions[k] for k in sorted(oof_predictions.keys())])
        
        self.meta_model = Ridge(alpha=0.5)
        oof_meta = cross_val_predict(self.meta_model, stack_features, y_full, cv=kf)
        self.meta_model.fit(stack_features, y_full)
        meta_mse = mean_squared_error(y_full, oof_meta)
        print(f"    Meta-learner OOF MSE: {meta_mse:.10f}")
        
        # Compare with simple average
        avg_pred = np.mean([oof_predictions[k] for k in oof_predictions], axis=0)
        avg_mse = mean_squared_error(y_full, avg_pred)
        print(f"    Simple Average MSE: {avg_mse:.10f}")
        
        # Final evaluation on validation set
        y_val_pred = self.predict(X_val, clip=False)
        
        self.train_metrics = {
            'train_mse': mean_squared_error(y_train, self.predict(X_train, clip=False)),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'meta_oof_mse': meta_mse,
            'n_models': len(self.models),
        }
        
        print(f"\n{'='*60}")
        print("Stacking Ensemble Training Complete!")
        print(f"{'='*60}")
        print(f"Models: {len(self.models)} ({', '.join(self.models.keys())})")
        print(f"Meta OOF MSE: {meta_mse:.6f}")
        print(f"Val MSE:      {self.train_metrics['val_mse']:.6f}")
        print(f"Val RMSE:     {self.train_metrics['val_rmse']:.6f}")
        print(f"Val MAE:      {self.train_metrics['val_mae']:.6f}")
        print(f"{'='*60}")
        
        return self.train_metrics
    
    def predict(self, X, clip=True):
        """Meta-learner stacking prediction."""
        # Get predictions from all base models
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        
        # Stack and predict with meta-learner
        stack_features = np.column_stack([preds[k] for k in sorted(preds.keys())])
        predictions = self.meta_model.predict(stack_features)
        
        if clip:
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def get_feature_importance(self):
        """Combined feature importance from tree models."""
        importances = []
        weights = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                imp = imp / imp.sum()  # Normalize
                importances.append(imp)
                # Weight by inverse MSE
                weights.append(1.0 / (self.model_mses.get(name, 1.0) + 1e-10))
        
        if not importances:
            return pd.DataFrame()
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        combined = sum(w * imp for w, imp in zip(weights, importances))
        
        names = self.feature_names or [f'feature_{i}' for i in range(len(combined))]
        
        return pd.DataFrame({
            'feature': names,
            'importance': combined,
        }).sort_values('importance', ascending=False)
    
    def save(self, path=None):
        """Save stacking ensemble to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'stacking_ensemble.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'meta_model': self.meta_model,
                'feature_names': self.feature_names,
                'train_metrics': self.train_metrics,
                'model_mses': self.model_mses,
            }, f)
        
        print(f"Stacking ensemble saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load stacking ensemble from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.models = data['models']
        instance.meta_model = data['meta_model']
        instance.feature_names = data['feature_names']
        instance.train_metrics = data.get('train_metrics', {})
        instance.model_mses = data.get('model_mses', {})
        
        print(f"Stacking ensemble loaded from {path}")
        return instance


        print(f"Ensemble loaded from {path}")
        return instance


class StackingEnsembleModel:
    """
    Multi-model stacking ensemble with MLP Neural Network and Ridge meta-learner.
    Models: XGBoost, LightGBM, CatBoost, ExtraTrees, MLPRegressor
    """
    
    MODEL_DIR = 'models'
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or self.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        self.models = {}
        self.meta_model = None
        self.feature_names = None
        self.train_metrics = {}
        self.model_mses = {}
        self.scaler = None  # For MLP
    
    def train(self, X_train, y_train, X_val, y_val, 
              feature_names=None, n_trials=25, timeout=300):
        """Train stacking ensemble with multiple models."""
        from sklearn.model_selection import cross_val_predict, KFold
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        
        # Try importing CatBoost
        try:
            from catboost import CatBoostRegressor
            HAS_CATBOOST = True
        except ImportError:
            HAS_CATBOOST = False
            print("CatBoost not available, skipping...")
        
        self.feature_names = feature_names
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\n" + "="*60)
        print("Training Multi-Model Stacking Ensemble with Neural Networks")
        print("="*60)
        
        # Combine train and val for OOF predictions
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        
        # Scale for MLP
        self.scaler = StandardScaler()
        X_full_scaled = self.scaler.fit_transform(X_full)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        oof_predictions = {}
        
        # ============ MODEL 1: XGBoost ============
        print("\n[1/5] Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,  # Lower learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        oof_xgb = cross_val_predict(xgb_model, X_full, y_full, cv=kf)
        xgb_model.fit(X_full, y_full)
        xgb_mse = mean_squared_error(y_full, oof_xgb)
        print(f"    XGBoost OOF MSE: {xgb_mse:.10f}")
        oof_predictions['xgb'] = oof_xgb
        self.models['xgb'] = xgb_model
        self.model_mses['xgb'] = xgb_mse
        
        # ============ MODEL 2: LightGBM ============
        print("\n[2/5] Training LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.03,  # Lower learning rate
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        oof_lgb = cross_val_predict(lgb_model, X_full, y_full, cv=kf)
        lgb_model.fit(X_full, y_full)
        lgb_mse = mean_squared_error(y_full, oof_lgb)
        print(f"    LightGBM OOF MSE: {lgb_mse:.10f}")
        oof_predictions['lgb'] = oof_lgb
        self.models['lgb'] = lgb_model
        self.model_mses['lgb'] = lgb_mse
        
        # ============ MODEL 3: CatBoost ============
        if HAS_CATBOOST:
            print("\n[3/5] Training CatBoost...")
            cat_model = CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.03,
                random_state=42,
                verbose=0,
                thread_count=-1
            )
            oof_cat = cross_val_predict(cat_model, X_full, y_full, cv=kf)
            cat_model.fit(X_full, y_full)
            cat_mse = mean_squared_error(y_full, oof_cat)
            print(f"    CatBoost OOF MSE: {cat_mse:.10f}")
            oof_predictions['catboost'] = oof_cat
            self.models['catboost'] = cat_model
            self.model_mses['catboost'] = cat_mse
        else:
            print("\n[3/5] Skipping CatBoost")
        
        # ============ MODEL 4: Neural Network (MLP) ============
        print("\n[4/5] Training Neural Network (MLP)...")
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        oof_mlp = cross_val_predict(mlp_model, X_full_scaled, y_full, cv=kf, n_jobs=-1)
        mlp_model.fit(X_full_scaled, y_full)
        mlp_mse = mean_squared_error(y_full, oof_mlp)
        print(f"    MLP OOF MSE: {mlp_mse:.10f}")
        oof_predictions['mlp'] = oof_mlp
        self.models['mlp'] = mlp_model
        self.model_mses['mlp'] = mlp_mse
        
        # ============ MODEL 5: ExtraTrees (Fast) ============
        print("\n[5/5] Training ExtraTrees...")
        et_model = ExtraTreesRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        oof_et = cross_val_predict(et_model, X_full, y_full, cv=kf)
        et_model.fit(X_full, y_full)
        et_mse = mean_squared_error(y_full, oof_et)
        print(f"    ExtraTrees OOF MSE: {et_mse:.10f}")
        oof_predictions['et'] = oof_et
        self.models['et'] = et_model
        self.model_mses['et'] = et_mse
        
        # ============ META-LEARNER: Ridge on OOF predictions ============
        print("\n[META] Training Ridge meta-learner...")
        
        # Collect predictions
        model_keys = sorted(oof_predictions.keys())
        stack_features = np.column_stack([oof_predictions[k] for k in model_keys])
        
        self.meta_model = Ridge(alpha=1.0) # slightly stronger regularization
        oof_meta = cross_val_predict(self.meta_model, stack_features, y_full, cv=kf)
        self.meta_model.fit(stack_features, y_full)
        meta_mse = mean_squared_error(y_full, oof_meta)
        print(f"    Meta-learner CV MSE: {meta_mse:.10f}")
        
        # Compare with simple average
        avg_pred = np.mean([oof_predictions[k] for k in oof_predictions], axis=0)
        avg_mse = mean_squared_error(y_full, avg_pred)
        print(f"    Simple Average MSE: {avg_mse:.10f}")
        
        # Validation evaluation (on original validation set, just for reference)
        y_val_pred = self.predict(X_val, clip=False)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        self.train_metrics = {
            'train_mse': mean_squared_error(y_train, self.predict(X_train, clip=False)),
            'val_mse': val_mse,
            'val_rmse': np.sqrt(val_mse),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'meta_oof_mse': meta_mse,
            'n_models': len(self.models),
        }
        
        print(f"\n{'='*60}")
        print("Stacking Ensemble Training Complete!")
        print(f"{'='*60}")
        print(f"Models: {len(self.models)} ({', '.join(self.models.keys())})")
        print(f"Meta OOF MSE: {meta_mse:.6f}")
        print(f"Val MSE:      {self.train_metrics['val_mse']:.6f} (reference)")
        print(f"{'='*60}")
        
        return self.train_metrics
    
    def predict(self, X, clip=True):
        """Meta-learner stacking prediction."""
        # Scale for MLP if needed
        X_mlp = self.scaler.transform(X) if self.scaler else X
        
        # Get predictions from all base models
        preds = {}
        for name, model in self.models.items():
            if name == 'mlp':
                preds[name] = model.predict(X_mlp)
            else:
                preds[name] = model.predict(X)
        
        # Stack and predict with meta-learner
        model_keys = sorted(self.models.keys())
        # Ensure only keys present in both
        valid_keys = [k for k in model_keys if k in preds]
        stack_features = np.column_stack([preds[k] for k in valid_keys])
        
        predictions = self.meta_model.predict(stack_features)
        
        if clip:
            predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def get_feature_importance(self):
        """Combined feature importance (excluding MLP/Non-tree models)."""
        importances = []
        weights = []
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                imp = imp / imp.sum()  # Normalize
                importances.append(imp)
                weights.append(1.0 / (self.model_mses.get(name, 1.0) + 1e-10))
        
        if not importances:
            return pd.DataFrame()
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        combined = sum(w * imp for w, imp in zip(weights, importances))
        
        names = self.feature_names or [f'feature_{i}' for i in range(len(combined))]
        
        return pd.DataFrame({
            'feature': names,
            'importance': combined,
        }).sort_values('importance', ascending=False)
    
    def save(self, path=None):
        """Save stacking ensemble to disk."""
        if path is None:
            path = os.path.join(self.model_dir, 'stacking_ensemble.pkl')
        
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'meta_model': self.meta_model,
                'feature_names': self.feature_names,
                'train_metrics': self.train_metrics,
                'model_mses': self.model_mses,
                'scaler': self.scaler
            }, f)
        
        print(f"Stacking ensemble saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load stacking ensemble from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.models = data['models']
        instance.meta_model = data['meta_model']
        instance.feature_names = data['feature_names']
        instance.train_metrics = data.get('train_metrics', {})
        instance.model_mses = data.get('model_mses', {})
        instance.scaler = data.get('scaler', None)
        
        print(f"Stacking ensemble loaded from {path}")
        return instance


def prepare_training_data(target_df: pd.DataFrame, 
                          pairwise_builder,
                          val_ratio: float = 0.2,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and validation data from target DataFrame.
    
    Args:
        target_df: DataFrame with src_user_id, dst_user_id, compatibility_score
        pairwise_builder: PairwiseFeatureBuilder instance
        val_ratio: Validation set ratio
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print(f"Building pairwise features for {len(target_df)} pairs...")
    
    X, y = pairwise_builder.build_pairwise_dataset(target_df, show_progress=True)
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state
    )
    
    print(f"Train: {len(X_train)} pairs, Val: {len(X_val)} pairs")
    
    return X_train, X_val, y_train, y_val


if __name__ == '__main__':
    # Test model training pipeline
    from feature_engineering import load_data, FeatureProcessor
    from embedding_generation import EmbeddingGenerator
    from pairwise_features import PairwiseFeatureBuilder
    
    # Load data
    train_df, test_df, target_df = load_data()
    
    # Process features
    processor = FeatureProcessor()
    processor.fit(train_df, test_df)
    train_participants = processor.process_dataframe(train_df)
    
    # Generate embeddings
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
    
    # Prepare training data
    X_train, X_val, y_train, y_val = prepare_training_data(target_df, builder)
    
    # Train model with reduced trials for testing
    model = CompatibilityModel()
    metrics = model.train(
        X_train, y_train, X_val, y_val,
        feature_names=builder.FEATURE_NAMES,
        tune=True,
        n_trials=20,  # Reduced for testing
        timeout=300
    )
    
    # Show feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance().to_string())
    
    # Save model
    model.save()
