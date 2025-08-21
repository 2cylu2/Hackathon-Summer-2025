import pandas as pd
import scanpy as sc
from pathlib import Path
import pickle
import numpy as np

# ---------------------------
# Paths
# ---------------------------
DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "hackathon"  # GEARS expects a folder
OUT_DIR.mkdir(exist_ok=True)

TRAIN_SINGLE_CSV = DATA_DIR / "train_single.csv"
H5AD_PATH = OUT_DIR / "hackathon.h5ad"

# ---------------------------
# 1) Load train_single.csv
# ---------------------------
train_single = pd.read_csv(TRAIN_SINGLE_CSV)

# First column = gene IDs
genes = train_single.iloc[:, 0].astype(str).tolist()
expr = train_single.iloc[:, 1:].copy()

# Collapse replicates if needed
expr.columns = [c.split(".")[0] for c in expr.columns]
expr_agg = expr.groupby(expr.columns, axis=1).mean()

# ---------------------------
# 2) Assign perturbation conditions
# ---------------------------
# GEARS expects single perturbations to have '+ctrl' in name
perturbations = [f"{c}+ctrl" for c in expr_agg.columns]

# ---------------------------
# 3) Build AnnData
# ---------------------------
X = expr_agg.T.values  # shape = samples x genes
adata = sc.AnnData(X)
adata.obs["condition"] = perturbations
adata.obs["cell_type"] = "hackathon_cell"
adata.var["gene_name"] = genes

# Use perturbation names as obs_names
adata.obs_names = perturbations

# ---------------------------
# 4) Save hackathon.h5ad
# ---------------------------
adata.write_h5ad(H5AD_PATH)
print(f"✅ Saved AnnData to {H5AD_PATH} with {adata.n_obs} samples and {adata.n_vars} genes")

# ---------------------------
# 5) Generate GEARS metadata
# ---------------------------

# Map perturbation name -> ID
node_map_pert = {pert: i for i, pert in enumerate(perturbations)}
with open(OUT_DIR / "node_map_pert.pkl", "wb") as f:
    pickle.dump(node_map_pert, f)

# Map gene name -> ID
node_map_gene = {gene: i for i, gene in enumerate(genes)}
with open(OUT_DIR / "node_map_gene.pkl", "wb") as f:
    pickle.dump(node_map_gene, f)

# Save empty edge_list (GEARS will use default connections)
edge_list = []
with open(OUT_DIR / "edge_list.pkl", "wb") as f:
    pickle.dump(edge_list, f)

print("✅ GEARS metadata files saved: node_map_pert.pkl, node_map_gene.pkl, edge_list.pkl")
print("Your hackathon.h5ad is now GEARS-ready!")
