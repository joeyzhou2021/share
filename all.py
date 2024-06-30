import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(x_file, y_file):
    X = pd.read_csv(x_file)
    y = pd.read_csv(y_file)
    return X, y

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_0_5 = f1_score(y_true, y_pred, beta=0.5)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f_0_5, roc_auc

def analyze_fingerprints(X, y):
    ecfp_columns = [col for col in X.columns if col.startswith('ecfp_fp_')]
    results = []

    y_true = y['measured outcome']
    y_pred = y['predicted outcome']

    for fingerprint in ecfp_columns:
        with_fingerprint = X[X[fingerprint] == 1].index
        without_fingerprint = X[X[fingerprint] == 0].index

        metrics_with = calculate_metrics(y_true[with_fingerprint], y_pred[with_fingerprint])
        metrics_without = calculate_metrics(y_true[without_fingerprint], y_pred[without_fingerprint])

        if metrics_with[3] > metrics_without[3]:  # Compare F0.5 scores
            results.append({
                'fingerprint': fingerprint,
                'with': metrics_with,
                'without': metrics_without
            })

    return results

def print_results(results):
    print("Fingerprints where F0.5 Score with fingerprint > F0.5 Score without fingerprint:")
    print("=" * 80)
    for result in results:
        print(f"Fingerprint: {result['fingerprint']}")
        print("Metrics with fingerprint:")
        print(f"  Accuracy: {result['with'][0]:.4f}")
        print(f"  Precision: {result['with'][1]:.4f}")
        print(f"  Recall: {result['with'][2]:.4f}")
        print(f"  F0.5 Score: {result['with'][3]:.4f}")
        print(f"  ROC AUC: {result['with'][4]:.4f}")
        print("Metrics without fingerprint:")
        print(f"  Accuracy: {result['without'][0]:.4f}")
        print(f"  Precision: {result['without'][1]:.4f}")
        print(f"  Recall: {result['without'][2]:.4f}")
        print(f"  F0.5 Score: {result['without'][3]:.4f}")
        print(f"  ROC AUC: {result['without'][4]:.4f}")
        print("-" * 80)

def main():
    x_file = 'aromatase_inhibitor_model_X.csv'
    y_file = 'aromatase_inhibitor_model_y.csv'

    X, y = load_data(x_file, y_file)
    results = analyze_fingerprints(X, y)
    print_results(results)

if __name__ == "__main__":
    main()
