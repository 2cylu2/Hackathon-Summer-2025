import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
PRED_DIR = Path("prediction")
PRED_DIR.mkdir(exist_ok=True)

# Normalize Training Data
def normalize_gene_expression(input_file, output_file, method='zscore'):
    """
    Normalize gene expression data from a CSV file
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to save normalized CSV file
    method (str): Normalization method - 'zscore', 'minmax', or 'log'
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file, index_col=0)
    
    # Extract the expression values (all rows, all columns except the first which is index)
    expression_data = df.values
    
    # Apply the selected normalization method
    if method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(expression_data)
    elif method == 'minmax':
        # Min-Max scaling to [0, 1] range
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(expression_data)
    elif method == 'log':
        # Log transformation (add 1 to avoid log(0))
        normalized_data = np.log2(expression_data + 1)
    else:
        raise ValueError("Method must be 'zscore', 'minmax', or 'log'")
    
    # Create a new DataFrame with normalized values
    normalized_df = pd.DataFrame(
        normalized_data,
        index=df.index,  # Keep the same gene names as index
        columns=df.columns  # Keep the same column names
    )
    
    # Save the normalized data to a new CSV file
    normalized_df.to_csv(output_file)
    
    print(f"Normalization complete. Saved to {output_file}")
    return normalized_df

# Example usage
if __name__ == "__main__":
    input_csv = "train_set.csv"
    output_csv = "normalized_gene_expression.csv"
    
    # Choose one of these normalization methods:
    # normalized_data = normalize_gene_expression(input_csv, output_csv, method='zscore')
    # normalized_data = normalize_gene_expression(input_csv, output_csv, method='minmax')
    normalized_data = normalize_gene_expression(input_csv, output_csv, method='log')
    
# Load data
train = pd.read_csv("normalized_gene_expression.csv")
test = pd.read_csv("test_set.csv", header=None, names=["perturbation"])

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

# ---------- Split into training and testing sets (80/20) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train_matrix, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing set: X_test={X_test.shape}, y_test={y_test.shape}")

if missing_singles_train:
    print(f"Note: Skipped {len(missing_singles_train)} genes with missing single profiles during training data creation.")

# ---------- Train a Gradient Boosting Model for Each Gene ----------
feature_names = [f"gA_{g}" for g in genes] + [f"gB_{g}" for g in genes]

X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print(f"Training {len(genes)} LightGBM models (one for each gene)...")
models = []
train_predictions = []
test_predictions = []

for i in tqdm(range(len(genes)), desc="Training models"):
    y_train_i = y_train[:, i]
    
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1) 
    
    lgbm.fit(X_train_df, y_train_i)
    models.append(lgbm)
    
    # Make predictions on training and test sets for evaluation
    train_pred_i = lgbm.predict(X_train_df)
    test_pred_i = lgbm.predict(X_test_df)
    
    train_predictions.append(train_pred_i)
    test_predictions.append(test_pred_i)

# Convert predictions to matrices
train_pred_matrix = np.column_stack(train_predictions)
test_pred_matrix = np.column_stack(test_predictions)

# ---------- Evaluate Model Performance ----------
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

# Calculate metrics for each gene
gene_metrics = []
for i in range(len(genes)):
    mse = mean_squared_error(y_test[:, i], test_pred_matrix[:, i])
    mae = mean_absolute_error(y_test[:, i], test_pred_matrix[:, i])
    r2 = r2_score(y_test[:, i], test_pred_matrix[:, i])
    rmsd = root_mean_squared_error(y_test[:, i], test_pred_matrix[:, i])
    
    gene_metrics.append({
        'gene': genes[i],
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmsd' : rmsd
    })

# Convert to DataFrame for better analysis
metrics_df = pd.DataFrame(gene_metrics)

# Print overall performance
print(f"\nOverall Performance on Test Set:")
print(f"Average MSE: {metrics_df['mse'].mean():.6f}")
print(f"Average MAE: {metrics_df['mae'].mean():.6f}")
print(f"Average R²: {metrics_df['r2'].mean():.6f}")
print(f"Median R²: {metrics_df['r2'].median():.6f}")
print(f"Median RMSE: {metrics_df['rmsd'].median():.6f}")


# ---------- Predict Test Set ----------
print("\nPredicting test set perturbations...")
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

# ---------- Additional validation checks ----------
expected_rows = len(genes) * len(test)
assert len(pred_df) == expected_rows, f"Expected {expected_rows} rows, got {len(pred_df)}"
assert list(pred_df.columns) == ["gene", "perturbation", "expression"]
assert pred_df["expression"].notna().all()

# ---------- Save predictions and metrics ----------
out_path = "prediction.csv"
pred_df.to_csv(out_path, index=False)
print(f"Saved {len(pred_df):,} predictions to {out_path}")

metrics_path = "model_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved model metrics to {metrics_path}")

if missing_singles_test:
    print(f"Note: missing single profiles for {len(missing_singles_test)} genes in test set, "
          f"used baseline fallback. Examples: {list(missing_singles_test)[:5]}")

# ---------- Plot performance distribution ----------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(metrics_df['r2'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('R² Score')
plt.ylabel('Number of Genes')
plt.title('Distribution of R² Scores Across Genes')

plt.subplot(1, 2, 2)
plt.scatter(metrics_df['mse'], metrics_df['r2'], alpha=0.6)
plt.xlabel('MSE')
plt.ylabel('R² Score')
plt.title('MSE vs R² Score')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel training and evaluation completed successfully!")