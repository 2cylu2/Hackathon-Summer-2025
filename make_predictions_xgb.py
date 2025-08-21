# make_predictions_xgb.py
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from scipy import sparse

DATA_DIR = Path("data")
PRED_DIR = Path("prediction")
PRED_DIR.mkdir(exist_ok=True)

# ---------------------------
# 1) Load data
# ---------------------------
print("Loading train_set.csv ...")
train = pd.read_csv(DATA_DIR / "train_set.csv", index_col=0)
print(f"Loaded train_set.csv: {train.shape[0]} genes × {train.shape[1]} perturbations")

test = pd.read_csv(DATA_DIR / "test_set.csv", header=None, names=["perturbation"])
print(f"Loaded test_set.csv: {test.shape[0]} perturbations")

genes = train.index.astype(str).tolist()
perturbations = train.columns.astype(str).tolist()

# Collapse train columns to canonical names (remove replicate suffixes)
canonical_perturbations = [c.split('.')[0] for c in train.columns]
train.columns = canonical_perturbations

# Now get unique perturbations for encoder
unique_perturbations = sorted(set(canonical_perturbations))
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
encoder.fit(np.array(unique_perturbations).reshape(-1, 1))
X_train_sparse = encoder.transform(np.array(canonical_perturbations).reshape(-1, 1))

# ---------------------------
# 3) Train one XGB model per gene
# ---------------------------
def train_gene(gene_idx):
    y = train.iloc[gene_idx].values
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=1  # safe inside parallel
    )
    model.fit(X_train_sparse, y)
    return model

print("Training XGBoost models per gene ...")
models = Parallel(n_jobs=-1, verbose=5)(
    delayed(train_gene)(i) for i in range(len(genes))
)
print(f"✅ Trained XGBoost models for {len(models)} genes")

# ---------------------------
# 4) Prepare test data
# ---------------------------
X_test_sparse = encoder.transform(test["perturbation"].values.reshape(-1, 1))

# ---------------------------
# 5) Generate predictions
# ---------------------------
print("Generating predictions ...")
pred_rows = []
for gene_idx, gene in enumerate(genes):
    y_pred = models[gene_idx].predict(X_test_sparse)
    for pert, val in zip(test["perturbation"], y_pred):
        pred_rows.append((gene, pert, float(val)))

pred_df = pd.DataFrame(pred_rows, columns=["gene", "perturbation", "expression"])

# ---------------------------
# 6) Save predictions
# ---------------------------
out_path = PRED_DIR / "prediction.csv"
pred_df.to_csv(out_path, index=False)
print(f"✅ Saved {len(pred_df):,} predictions to {out_path}")
