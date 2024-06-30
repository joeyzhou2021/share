import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    cm = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f_0_5, roc_auc, cm

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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

    return results, len(ecfp_columns)

def print_results(results, total_fingerprints):
    print(f"Number of ECFP fingerprints meeting the requirement: {len(results)} out of {total_fingerprints}")
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
        
        # Plot confusion matrices
        plot_confusion_matrix(result['with'][5], f"Confusion Matrix with {result['fingerprint']}")
        plot_confusion_matrix(result['without'][5], f"Confusion Matrix without {result['fingerprint']}")

def main():
    x_file = 'aromatase_inhibitor_model_X.csv'
    y_file = 'aromatase_inhibitor_model_y.csv'

    X, y = load_data(x_file, y_file)
    results, total_fingerprints = analyze_fingerprints(X, y)
    print_results(results, total_fingerprints)

if __name__ == "__main__":
    main()



import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(x_file, y_file):
    X = pd.read_csv(x_file)
    y = pd.read_csv(y_file)
    return X, y

def preprocess_data(X):
    # Standardize all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def perform_tsne(X_scaled):
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    return X_tsne

def get_prediction_categories(y_true, y_pred):
    categories = []
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            categories.append('True Positive')
        elif true == 0 and pred == 0:
            categories.append('True Negative')
        elif true == 0 and pred == 1:
            categories.append('False Positive')
        else:
            categories.append('False Negative')
    return categories

def plot_tsne(X_tsne, categories):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=categories, palette='deep')
    plt.title('t-SNE Plot of All Features')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Prediction Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    x_file = 'aromatase_inhibitor_model_X.csv'
    y_file = 'aromatase_inhibitor_model_y.csv'

    # Load data
    X, y = load_data(x_file, y_file)

    # Preprocess data
    X_scaled = preprocess_data(X)
    print(f"Total number of features used: {X.shape[1]}")

    # Perform t-SNE
    X_tsne = perform_tsne(X_scaled)

    # Get prediction categories
    y_true = y['measured outcome']
    y_pred = y['predicted outcome']
    categories = get_prediction_categories(y_true, y_pred)

    # Plot t-SNE
    plot_tsne(X_tsne, categories)

if __name__ == "__main__":
    main()
