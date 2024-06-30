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

def analyze_fingerprint(X, y, fingerprint):
    # Split data based on fingerprint presence
    with_fingerprint = X[X[fingerprint] == 1].index
    without_fingerprint = X[X[fingerprint] == 0].index
    
    # Calculate metrics for both groups
    y_true = y['measured outcome']
    y_pred = y['predicted outcome']
    
    metrics_with = calculate_metrics(y_true[with_fingerprint], y_pred[with_fingerprint])
    metrics_without = calculate_metrics(y_true[without_fingerprint], y_pred[without_fingerprint])
    
    return metrics_with, metrics_without

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def main():
    X, y = load_data('aromatase_inhibitor_model_X.csv', 'aromatase_inhibitor_model_y.csv')
    
    fingerprint = 'ecfp_3651'  # Example fingerprint, replace with the one you want to analyze
    
    metrics_with, metrics_without = analyze_fingerprint(X, y, fingerprint)
    
    print(f"Metrics for molecules with {fingerprint}:")
    print(f"Accuracy: {metrics_with[0]:.4f}")
    print(f"Precision: {metrics_with[1]:.4f}")
    print(f"Recall: {metrics_with[2]:.4f}")
    print(f"F0.5 Score: {metrics_with[3]:.4f}")
    print(f"ROC AUC: {metrics_with[4]:.4f}")
    
    print(f"\nMetrics for molecules without {fingerprint}:")
    print(f"Accuracy: {metrics_without[0]:.4f}")
    print(f"Precision: {metrics_without[1]:.4f}")
    print(f"Recall: {metrics_without[2]:.4f}")
    print(f"F0.5 Score: {metrics_without[3]:.4f}")
    print(f"ROC AUC: {metrics_without[4]:.4f}")
    
    plot_confusion_matrix(metrics_with[5], f"Confusion Matrix (with {fingerprint})")
    plot_confusion_matrix(metrics_without[5], f"Confusion Matrix (without {fingerprint})")

if __name__ == "__main__":
    main()


import pandas as pd
import pickle

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def reorder_dataframe(csv_file, pkl_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Load the PKL file
    pkl_data = load_pkl(pkl_file)
    
    # Assuming the PKL file contains a pandas Series or a list
    # If it's a DataFrame, you might need to extract the relevant column
    if isinstance(pkl_data, pd.DataFrame):
        pkl_data = pkl_data.iloc[:, 0]  # Take the first column
    
    # Create a dictionary for quick lookup of indices
    index_map = {code: i for i, code in enumerate(pkl_data)}
    
    # Create a new column with the desired order
    df['new_order'] = df['BCS-code'].map(index_map)
    
    # Sort the DataFrame based on the new order
    df_sorted = df.sort_values('new_order')
    
    # Drop the temporary column
    df_final = df_sorted.drop('new_order', axis=1)
    
    return df_final

# Example usage
csv_file = 'input.csv'
pkl_file = 'order.pkl'

# Get the reordered DataFrame
reordered_df = reorder_dataframe(csv_file, pkl_file)

# Now you can use reordered_df for further processing
# For example:
print(reordered_df.head())

# You can perform more operations on reordered_df as needed
# For instance:
# some_result = perform_some_analysis(reordered_df)
# visualize_data(reordered_df)
# etc.
