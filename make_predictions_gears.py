# make_predictions_gears.py
import pandas as pd
import numpy as np
from pathlib import Path
import scanpy as sc
from gears import PertData, GEARS

# ---------------------------
# 0) Paths
# ---------------------------
DATA_DIR = Path("data")
HACKATHON_H5AD = DATA_DIR / "hackathon" / "hackathon.h5ad"
TEST_CSV = DATA_DIR / "test_set.csv"
PRED_DIR = Path("prediction")
PRED_DIR.mkdir(exist_ok=True)

# ---------------------------
# 1) Load hackathon.h5ad into scanpy
# ---------------------------
print(f"Loading {HACKATHON_H5AD}...")
adata = sc.read_h5ad(HACKATHON_H5AD)
print(f"AnnData loaded: {adata.n_obs} samples, {adata.n_vars} genes")
print("Sample conditions (first 10):", adata.obs["condition"].unique()[:10])

# ---------------------------
# 2) Wrap with PertData and process custom dataset
# ---------------------------
pert_data = PertData(DATA_DIR / "hackathon")  # folder where processed data will be saved
pert_data.new_data_process(dataset_name="hackathon", adata=adata)  # creates all internal structures

# Load the processed data into PertData
pert_data.load(data_path=DATA_DIR / "hackathon")  # folder with processed data

# Prepare simulation split and dataloaders
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=32, test_batch_size=128)
print("✅ PertData prepared with splits and dataloaders.")

# ---------------------------
# 3) Identify single perturbations for training
# ---------------------------
single_mask = adata.obs["condition"].str.contains(r"\+ctrl|^ctrl\+")
single_adata = adata[single_mask].copy()

if single_adata.n_obs == 0:
    raise ValueError(
        "No single perturbation cells found in hackathon.h5ad! "
        "Make sure conditions include '+ctrl' or 'ctrl+' suffix."
    )

X_single = pd.DataFrame(
    single_adata.X,
    index=single_adata.obs_names,
    columns=single_adata.var_names,
)

print(f"Using {X_single.shape[0]} single-perturbation samples for training.")

# ---------------------------
# 4) Initialize and train GEARS
# ---------------------------
model = GEARS(pert_data=pert_data)
model.fit(X_single, X_single, verbose=True)
print("✅ GEARS model trained on single perturbations.")

# ---------------------------
# 5) Load test double perturbations
# ---------------------------
test_df = pd.read_csv(TEST_CSV, header=None, names=["perturbation"])
genes = adata.var_names.tolist()

# Convert single perturbations to dict for fast lookup
single_dict = {cond: X_single.loc[cond].values for cond in X_single.index}

rows = []
missing_singles = set()

for pert in test_df["perturbation"].astype(str):
    if "+" not in pert:
        raise ValueError(f"Unexpected perturbation format: {pert}")
    gA, gB = pert.split("+", 1)

    vecA = single_dict.get(gA)
    vecB = single_dict.get(gB)

    # Fallback to median if a single is missing
    if vecA is None:
        missing_singles.add(gA)
        vecA = X_single.median(axis=0).values
    if vecB is None:
        missing_singles.add(gB)
        vecB = X_single.median(axis=0).values

    combined_input = pd.DataFrame([vecA + vecB], columns=genes)
    pred_vec = model.predict(combined_input)  # shape = 1 x genes

    for gid, val in zip(genes, pred_vec[0]):
        rows.append((gid, pert, float(val)))

# ---------------------------
# 6) Save predictions
# ---------------------------
pred_df = pd.DataFrame(rows, columns=["gene", "perturbation", "expression"])
out_path = PRED_DIR / "prediction.csv"
pred_df.to_csv(out_path, index=False)
print(f"✅ Saved {len(pred_df):,} predictions to {out_path}")

if missing_singles:
    print(
        f"Note: missing single profiles for {len(missing_singles)} perturbations, "
        f"used median fallback. Examples: {list(missing_singles)[:5]}"
    )
