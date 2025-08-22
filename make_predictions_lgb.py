import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm

DATA_DIR = Path("data")
PRED_DIR = Path("prediction")
PRED_DIR.mkdir(exist_ok=True)


# ---------- Load ----------
train = pd.read_csv(DATA_DIR / "train_set.csv")
test = pd.read_csv(DATA_DIR / "test_set.csv", header=None, names=["perturbation"])

print("Train shape:", train.shape)
print("Test shape:", test.shape)

if "Unnamed: 0" not in train.columns:
    raise ValueError("Expected first column 'Unnamed: 0' with gene IDs.")
genes = train["Unnamed: 0"].astype(str).tolist()

expr = train.drop(columns=["Unnamed: 0"]).copy()


# ---------- Collapse replicates ----------
canonical = [c.split('.')[0] for c in expr.columns]
expr.columns = canonical
expr_agg = expr.groupby(level=0, axis=1).mean()


# ---------- Identify single-perturbation columns & baseline ----------
single_cols = [c for c in expr_agg.columns if c.endswith("+ctrl") or c.startswith("ctrl+")]

if len(single_cols) == 0:
    print("Warning: couldn't find single-perturbation columns by '+ctrl'. Using median as baseline.")
    baseline_vec = expr_agg.median(axis=1).values
else:
    baseline_vec = expr_agg[single_cols].median(axis=1).values

def get_single_vector(gene_id: str):
    key1 = f"{gene_id}+ctrl"
    key2 = f"ctrl+{gene_id}"
    if key1 in expr_agg.columns:
        return expr_agg[key1].values
    if key2 in expr_agg.columns:
        return expr_agg[key2].values
    return None


# ---------- Prepare Training Data for Gradient Boosting ----------
print("Preparing training data for the machine learning model...")
double_train_cols = [c for c in expr_agg.columns if "+ctrl" not in c and "ctrl+" not in c]

X_train_list = []
y_train_list = []
missing_singles_train = set()

for pert in double_train_cols:
    if "+" not in pert:
        continue
    gA, gB = pert.split("+", 1)

    vecA = get_single_vector(gA)
    vecB = get_single_vector(gB)

    if vecA is None or vecB is None:
        if vecA is None: missing_singles_train.add(gA)
        if vecB is None: missing_singles_train.add(gB)
        continue

    X_train_list.append(np.concatenate([vecA, vecB]))
    y_train_list.append(expr_agg[pert].values)

X_train = np.array(X_train_list)
y_train_matrix = np.array(y_train_list)

print(f"Created training set with shape X: {X_train.shape}, y: {y_train_matrix.shape}")
if missing_singles_train:
    print(f"Note: Skipped {len(missing_singles_train)} genes with missing single profiles during training data creation.")


# ---------- Train a Gradient Boosting Model for Each Gene ----------
feature_names = [f"gA_{g}" for g in genes] + [f"gB_{g}" for g in genes]

X_train_df = pd.DataFrame(X_train, columns=feature_names)


print(f"Training {len(genes)} LightGBM models (one for each gene)...")
models = []
for i in tqdm(range(len(genes)), desc="Training models"):
    y_train_i = y_train_matrix[:, i]
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1) 
    
    lgbm.fit(X_train_df, y_train_i)

    models.append(lgbm)


# ---------- Predict Each Test Pair Using the Trained Models ----------
print("Predicting test set perturbations...")
rows = []
missing_singles_test = set()

for pert in test["perturbation"].astype(str):
    gA, gB = pert.split("+", 1)

    vecA = get_single_vector(gA)
    vecB = get_single_vector(gB)

    if vecA is None:
        missing_singles_test.add(gA)
        vecA = baseline_vec
    if vecB is None:
        missing_singles_test.add(gB)
        vecB = baseline_vec

    x_test_sample = np.concatenate([vecA, vecB]).reshape(1, -1)

    x_test_df = pd.DataFrame(x_test_sample, columns=feature_names)
    
    pred_vec = np.array([model.predict(x_test_df)[0] for model in models])

    for gid, val in zip(genes, pred_vec):
        rows.append((gid, pert, float(val)))


pred_df = pd.DataFrame(rows, columns=["gene", "perturbation", "expression"])


# ---------- Checks ----------
expected_rows = len(genes) * len(test)
assert len(pred_df) == expected_rows, f"Expected {expected_rows} rows, got {len(pred_df)}"
assert list(pred_df.columns) == ["gene", "perturbation", "expression"]
assert pred_df["expression"].notna().all()


# ---------- Save ----------
out_path = PRED_DIR / "prediction.csv"
pred_df.to_csv(out_path, index=False)
print(f"Saved {len(pred_df):,} predictions to {out_path}")

if missing_singles_test:
    print(f"Note: missing single profiles for {len(missing_singles_test)} genes in test set, "
          f"used baseline fallback. Examples: {list(missing_singles_test)[:5]}")