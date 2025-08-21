import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import trim_mean  # Add this import

DATA_DIR = Path("data")
PRED_DIR = Path("prediction")
PRED_DIR.mkdir(exist_ok=True)

# ---------- 1) Load ----------
train = pd.read_csv(DATA_DIR / "train_set.csv")
# test_set.csv has no header; ensure we don't treat first row as a header
test = pd.read_csv(DATA_DIR / "test_set.csv", header=None, names=["perturbation"])

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# gene IDs are in the first column named like 'Unnamed: 0'
if "Unnamed: 0" not in train.columns:
    raise ValueError("Expected first column 'Unnamed: 0' with gene IDs.")
genes = train["Unnamed: 0"].astype(str).tolist()

# expression matrix with columns = perturbations, rows = genes
expr = train.drop(columns=["Unnamed: 0"]).copy()

# ---------- 2) Collapse replicates ----------
# Some columns look like 'g0843+ctrl.39'. We collapse to 'g0843+ctrl' by averaging replicates.
canonical = [c.split('.')[0] for c in expr.columns]  # drop everything after first '.'
expr.columns = canonical

# Average columns with the same canonical name
expr_agg = expr.groupby(level=0, axis=1).mean()

# ---------- 3) Identify single-perturbation columns & baseline ----------
# Single columns typically end with '+ctrl' or start with 'ctrl+'
single_cols = [c for c in expr_agg.columns if c.endswith("+ctrl") or c.startswith("ctrl+")]

if len(single_cols) == 0:
    print("Warning: couldn't find single-perturbation columns by '+ctrl'. "
          "Using trimmed mean across ALL columns as baseline.")
    baseline_vec = trim_mean(expr_agg.values, proportiontocut=0.1, axis=1)  # shape (1000,)
else:
    baseline_vec = trim_mean(expr_agg[single_cols].values, proportiontocut=0.1, axis=1)  # shape (1000,)

# Helper to fetch the single profile for a gene (g####)
def get_single_vector(gene_id: str):
    key1 = f"{gene_id}+ctrl"
    key2 = f"ctrl+{gene_id}"
    if key1 in expr_agg.columns:
        return expr_agg[key1].values  # shape (1000,)
    if key2 in expr_agg.columns:
        return expr_agg[key2].values
    return None  # missing single profile

# ---------- 4) Predict each test pair ----------
rows = []
missing_singles = set()

for pert in test["perturbation"].astype(str):
    if "+" not in pert:
        raise ValueError(f"Unexpected perturbation format: {pert}")
    gA, gB = pert.split("+", 1)

    vecA = get_single_vector(gA)
    vecB = get_single_vector(gB)

    # Fallbacks if any single is missing: use baseline (neutral effect)
    if vecA is None:
        missing_singles.add(gA)
        vecA = baseline_vec
    if vecB is None:
        missing_singles.add(gB)
        vecB = baseline_vec

    # Additive baseline: A + B - baseline
    pred_vec = vecA + vecB - baseline_vec  # shape (1000,)

    # Emit rows in required long format
    for gid, val in zip(genes, pred_vec):
        rows.append((gid, pert, float(val)))

pred_df = pd.DataFrame(rows, columns=["gene", "perturbation", "expression"])

# ---------- 5) Integrity checks ----------
expected_rows = len(genes) * len(test)
assert len(pred_df) == expected_rows, f"Expected {expected_rows} rows, got {len(pred_df)}"
assert list(pred_df.columns) == ["gene", "perturbation", "expression"]
assert pred_df["expression"].notna().all()

# ---------- 6) Save ----------
out_path = PRED_DIR / "prediction.csv"
pred_df.to_csv(out_path, index=False)
print(f"Saved {len(pred_df):,} predictions to {out_path}")

if missing_singles:
    print(f"Note: missing single profiles for {len(missing_singles)} genes, "
          f"used baseline fallback. Examples: {list(missing_singles)[:5]}")


