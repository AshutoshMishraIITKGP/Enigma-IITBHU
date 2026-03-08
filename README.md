# Enigma CodeFest - Compatibility Prediction Solution

## Overview
This repository contains a high-performance solution for the Enigma CodeFest Reciprocal Recommendation challenge. The goal is to predict compatibility scores (0-1) between pairs of professional profiles.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
```bash
python run_steps.py all
```

### 3. Run Individual Steps (for debugging or retraining)
| Step | Command | Description |
|------|---------|-------------|
| 1 | `python run_steps.py step1` | Fine-tune SBERT on compatibility pairs |
| 2 | `python run_steps.py step2` | Train learned categorical embeddings |
| 2b | `python run_steps.py step2b` | Train Graph SVD (latent features) |
| 3 | `python run_steps.py step3` | Generate semantic embeddings |
| 4 | `python run_steps.py step4` | Train Stacking Ensemble model |
| 5 | `python run_steps.py step5` | Generate `submission.csv` |

---

## Architecture

### Feature Engineering (74 Features)
The solution combines **content-based** and **collaborative filtering** approaches:

1. **Structured Features (16):** Age difference, seniority gap, industry/location/role match, company size ratio.
2. **Interest Overlap (4):** Jaccard similarity, overlap count, individual interest counts.
3. **Semantic Features (4):** Cosine similarity between SBERT embeddings of Objectives and Constraints (Obj-Obj, Obj-Con, Con-Obj, Con-Con).
4. **Feature Crosses (8):** Interaction terms (e.g., `industry_match × interests_jaccard`).
5. **Attention Features (6):** Cross-attention alignment scores between text embeddings.
6. **Learned Categorical Embeddings (4):** Similarity of Industry, Role, Location, Company embeddings trained end-to-end.
7. **Contextual Semantic Features (10):** Semantic similarity modulated by professional context.
8. **Raw Text Set Features (19):** Jaccard, Dice, Overlap coefficients on parsed text sets (Business Interests, Objectives, Constraints).
9. **Graph SVD Features (3):** Latent factors from Matrix Factorization on the interaction graph (`svd_dot`, `svd_cosine`, `svd_euclidean`).

### Model: Advanced Stacking Ensemble
The final model is a **Stacking Ensemble** with:
- **Base Models:**
  - XGBoost
  - LightGBM
  - CatBoost
  - ExtraTrees
  - MLP (Neural Network)
- **Meta-Learner:** Ridge Regression

Out-of-fold (OOF) predictions from each base model are stacked and fed into the Ridge meta-learner.

---

## Workflow Story (How We Got Here)

### Phase 1: Baseline
- Started with basic structured features (age diff, industry match) and XGBoost.
- Added feature crosses to capture interactions.

### Phase 2: Semantic Understanding
- Fine-tuned **Sentence-BERT** (`all-mpnet-base-v2`) on compatibility pairs using contrastive learning.
- Generated 768-dim embeddings for Business Objectives and Constraints.
- Added cosine similarities between embeddings as features.

### Phase 3: Advanced Text Features
- Implemented cross-attention alignment between text embeddings.
- Added asymmetric conflict signals (e.g., `objectives(A) vs constraints(B)`).

### Phase 4: Learned Embeddings
- Trained end-to-end categorical embeddings for Industry, Role, Location, Company using compatibility labels.

### Phase 5: Notebook-Inspired Features
- Added raw text set features (Jaccard, Dice, Overlap) directly on parsed text fields.
- Implemented cross-category features (e.g., `|BI_A ∩ BO_B|`).

### Phase 6: Graph/Latent Features (Key Insight)
- Constructed the interaction graph from `target.csv`.
- Trained **TruncatedSVD** to extract 32-dim latent vectors for each user.
- Added `svd_cosine` as a feature — **this became the #1 most important feature (0.17 importance)**.

### Phase 7: Stacking Ensemble
- Replaced simple weighted average with a **Stacking Ensemble**.
- Used 5-fold cross-validation to generate OOF predictions for the Ridge meta-learner.

### Final Training
Training was performed with an **80/20 train/validation split**:
- **Training set:** 288,000 pairs
- **Validation set:** 72,000 pairs

**Final Validation Results (from last run):**
| Model | OOF MSE |
|-------|---------|
| XGBoost | 0.0000617 |
| LightGBM | 0.0000691 |
| CatBoost | 0.0000951 |
| **MLP** | **0.0000301** |
| ExtraTrees | 0.0000847 |
| **Meta-learner (Ridge)** | **0.0000294** |

The stacking ensemble achieved significant improvement by combining diverse models and leveraging the powerful graph-based features.

---

## File Structure
```
├── run_steps.py           # Main orchestrator script
├── feature_engineering.py # Data loading and feature processing
├── pairwise_features.py   # Pairwise feature construction (74 features)
├── model_training.py      # StackingEnsembleModel definition
├── svd_features.py        # Graph SVD feature extraction
├── embedding_generation.py # SBERT embedding generation
├── finetune_sbert.py      # SBERT fine-tuning script
├── learned_embeddings.py  # Learned categorical embeddings
├── attention_features.py  # Cross-attention features
├── inference.py           # Prediction pipeline
├── requirements.txt       # Dependencies
├── submission.csv         # Final predictions
├── train.xlsx             # Training data (not included in zip)
├── test.xlsx              # Test data (not included in zip)
└── models/                # Saved models and checkpoints
```

---

## Key Takeaways
1. **Graph features are powerful:** `svd_cosine` became the most important feature, demonstrating that collaborative filtering signals complement content-based features.
2. **Diversity matters:** The stacking ensemble combines gradient boosting (XGB, LGB, CatBoost), neural networks (MLP), and random forests (ExtraTrees) for robust predictions.
3. **Fine-tuned SBERT helps:** Domain-specific fine-tuning improved semantic similarity features.
