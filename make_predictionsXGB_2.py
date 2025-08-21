import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import re

def parse_perturbation_labels(perturbation_labels):
    """Parse perturbation labels to extract which genes are perturbed"""
    samples = []
    for label in perturbation_labels:
        if label.lower() == 'control':
            samples.append((label, []))
        else:
            genes = re.split(r'[+_\-]', label)
            genes = [g.strip() for g in genes if g.strip()]
            samples.append((label, genes))
    return samples

def create_feature_matrix(perturbation_samples, gene_names):
    """Create feature matrix where each row represents a sample"""
    n_samples = len(perturbation_samples)
    n_genes = len(gene_names)
    
    X = np.zeros((n_samples, n_genes))
    sample_names = []
    
    for i, (sample_name, perturbed_genes) in enumerate(perturbation_samples):
        sample_names.append(sample_name)
        for gene in perturbed_genes:
            if gene in gene_names:
                gene_idx = gene_names.index(gene)
                X[i, gene_idx] = 1
    
    return X, sample_names

def load_and_prepare_data(filename):
    """Load the CSV file and prepare features and targets"""
    df = pd.read_csv(filename, index_col=0)
    
    perturbation_labels = df.columns.tolist()
    gene_names = df.index.tolist()
    expression_matrix = df.values
    
    print(f"Data loaded: {len(gene_names)} genes × {len(perturbation_labels)} samples")
    
    perturbation_samples = parse_perturbation_labels(perturbation_labels)
    X, sample_names = create_feature_matrix(perturbation_samples, gene_names)
    y = expression_matrix.T
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target matrix shape: {y.shape}")
    
    return X, y, gene_names, sample_names

def train_xgboost_model(X, y, test_size=0.2):
    """Train XGBoost model without parallel processing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Use single-threaded XGBoost
    base_model = xgb.XGBRegressor(
        n_estimators=100,  # Reduced for faster training
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,  # Force single thread
        #early_stopping_rounds=20
    )
    
    # Use single-threaded MultiOutputRegressor
    model = MultiOutputRegressor(base_model, n_jobs=1)
    
    print("Training model (single-threaded)...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return model, X_test, y_test, y_pred

def predict_double_perturbation(model, gene_a, gene_b, gene_names):
    """Predict gene expression for a double perturbation"""
    feature_vector = np.zeros(len(gene_names))
    
    try:
        idx_a = gene_names.index(gene_a)
        idx_b = gene_names.index(gene_b)
        feature_vector[idx_a] = 1
        feature_vector[idx_b] = 1
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    predictions = model.predict(feature_vector.reshape(1, -1))[0]
    result = {gene: expr for gene, expr in zip(gene_names, predictions)}
    
    return result

# Main execution
def main():
    X, y, gene_names, sample_names = load_and_prepare_data('normalized_gene_expression.csv')
    model, X_test, y_test, y_pred = train_xgboost_model(X, y)
    
    # Example prediction
    gene_a = 'g0037'
    gene_b = 'g0083,'
    
    predictions = predict_double_perturbation(model, gene_a, gene_b, gene_names)
    
    if predictions:
        print(f"\nPredicted expression for {gene_a}+{gene_b}:")
        sorted_predictions = sorted(predictions.items(), key=lambda x: abs(x[1]), reverse=True)
        for gene, expr in sorted_predictions[:10]:
            print(f"{gene}: {expr:.4f}")
    
    return model, gene_names

if __name__ == "__main__":
    model, gene_names = main()