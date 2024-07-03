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


def get_false_predictions(y_true, y_pred):
    categories = []
    indices = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 0 and pred == 1:
            categories.append('False Positive')
            indices.append(i)
        elif true == 1 and pred == 0:
            categories.append('False Negative')
            indices.append(i)
    return categories, indices

def plot_tsne(X_tsne, categories):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=categories, palette='Set1')
    plt.title('t-SNE Plot of False Predictions')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(title='Prediction Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


-----
import pandas as pd
import pickle

def load_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)

def filter_split_and_reorder_dataframe(csv_file, pkl_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Load the PKL file
    pkl_data = load_pkl(pkl_file)
    
    # Assuming the PKL file contains a pandas Series or a list
    # If it's a DataFrame, you might need to extract the relevant column
    if isinstance(pkl_data, pd.DataFrame):
        pkl_data = pkl_data.iloc[:, 0]  # Take the first column
    
    # Convert pkl_data to a set for faster lookup
    pkl_set = set(pkl_data)
    
    # Filter the DataFrame to keep only rows with BCS-codes in the PKL file
    df_in_pkl = df[df['BCS-code'].isin(pkl_set)].copy()
    
    # Filter the DataFrame to keep only rows with BCS-codes NOT in the PKL file
    df_not_in_pkl = df[~df['BCS-code'].isin(pkl_set)].copy()
    
    # Create a dictionary for quick lookup of indices
    index_map = {code: i for i, code in enumerate(pkl_data)}
    
    # Create a new column with the desired order for df_in_pkl
    df_in_pkl.loc[:, 'new_order'] = df_in_pkl['BCS-code'].map(index_map)
    
    # Sort the DataFrame based on the new order
    df_in_pkl_sorted = df_in_pkl.sort_values('new_order')
    
    # Drop the temporary column
    df_in_pkl_final = df_in_pkl_sorted.drop('new_order', axis=1)
    
    # Reset the index for both dataframes
    df_in_pkl_final = df_in_pkl_final.reset_index(drop=True)
    df_not_in_pkl = df_not_in_pkl.reset_index(drop=True)
    
    return df_in_pkl_final, df_not_in_pkl

csv_file = 'input.csv'
pkl_file = 'order.pkl'

# Get the filtered, split, and reordered DataFrames
df_in_pkl, df_not_in_pkl = filter_split_and_reorder_dataframe(csv_file, pkl_file)

print(f"Number of rows in DataFrame with BCS-codes in PKL: {len(df_in_pkl)}")
print(f"Number of rows in DataFrame with BCS-codes NOT in PKL: {len(df_not_in_pkl)}")

print("\nFirst few rows of DataFrame with BCS-codes in PKL:")
print(df_in_pkl.head())

print("\nFirst few rows of DataFrame with BCS-codes NOT in PKL:")
print(df_not_in_pkl.head())
--------------

import pandas as pd

# Load the pickle files
file1_path = 'path_to_file1.pkl'
file2_path = 'path_to_file2.pkl'

df1 = pd.read_pickle(file1_path)
df2 = pd.read_pickle(file2_path)

# Ensure both DataFrames have the same columns in the same order
df1 = df1.sort_index(axis=1)
df2 = df2.sort_index(axis=1)

# Find common rows
common_rows = pd.merge(df1, df2, how='inner')

# Display the number of common rows
print(f"Number of common rows: {len(common_rows)}")

# display the common rows
# print(common_rows)


---------
import pandas as pd

# Load the pickle files
df1 = pd.read_pickle('file1.pkl')
df2 = pd.read_pickle('file2.pkl')

# Ensure both dataframes have the same columns
common_columns = df1.columns.intersection(df2.columns)
df1 = df1[common_columns]
df2 = df2[common_columns]

# Compare the dataframes
same_data = (df1 == df2).all(axis=1)

# Count the number of rows with the same data
num_same_rows = same_data.sum()

print(f"Number of rows with the same data: {num_same_rows}")
print(f"Total number of rows: {len(df1)}")
print(f"Percentage of matching rows: {(num_same_rows / len(df1)) * 100:.2f}%")


---------------
from rdkit import Chem
import pandas as pd

# Define the path to your SDF file
sdf_file_path = 'path_to_your_file.sdf'

# Read the SDF file
supplier = Chem.SDMolSupplier(sdf_file_path)

# Extract data from SDF
data = []
for mol in supplier:
    if mol is not None:
        props = mol.GetPropsAsDict()
        data.append(props)

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

------
import pandas as pd

# Load the DataFrames
df1 = pd.read_pickle('file1.pkl')
df2 = pd.read_pickle('file2.pkl')

# Specify the columns to compare
column1 = 'column1'  # Replace with the actual column name from df1
column2 = 'column2'  # Replace with the actual column name from df2

# Combine the specified columns from both DataFrames
combined_df = pd.DataFrame({
    'df1_column': df1[column1],
    'df2_column': df2[column2]
})

# Remove rows where either column has a missing value
combined_df = combined_df.dropna()

# Count matching rows
matching_rows = (combined_df['df1_column'] == combined_df['df2_column']).sum()

# Calculate total rows and percentage
total_rows = len(combined_df)
matching_percentage = (matching_rows / total_rows) * 100 if total_rows > 0 else 0

# Print results
print(f"Matching rows: {matching_rows}")
print(f"Total rows compared: {total_rows}")
print(f"Matching percentage: {matching_percentage:.2f}%")

# Optional: Display the first few matching and non-matching rows
print("\nSample of matching rows:")
print(combined_df[combined_df['df1_column'] == combined_df['df2_column']].head())

print("\nSample of non-matching rows:")
print(combined_df[combined_df['df1_column'] != combined_df['df2_column']].head())

--------------------------------------
import pandas as pd

# Load the DataFrames
df1 = pd.read_pickle('file1.pkl')
df2 = pd.read_pickle('file2.pkl')

# Specify the columns to compare
column1 = 'column1'  # Replace with the actual column name from df1
column2 = 'column2'  # Replace with the actual column name from df2

# Count missing values
missing_df1 = df1[column1].isna().sum()
missing_df2 = df2[column2].isna().sum()

print(f"Missing values in df1[{column1}]: {missing_df1}")
print(f"Missing values in df2[{column2}]: {missing_df2}")

# Remove missing values for comparison
values1 = df1[column1].dropna().values
values2 = df2[column2].dropna().values

# Convert to sets for unordered comparison
set1 = set(values1)
set2 = set(values2)

# Find common values
common_values = set1.intersection(set2)

# Calculate statistics
total_unique_values = len(set1.union(set2))
matching_values = len(common_values)
matching_percentage = (matching_values / total_unique_values) * 100 if total_unique_values > 0 else 0

# Print results
print(f"\nUnique values in df1[{column1}]: {len(set1)}")
print(f"Unique values in df2[{column2}]: {len(set2)}")
print(f"Matching unique values: {matching_values}")
print(f"Total unique values across both columns: {total_unique_values}")
print(f"Matching percentage: {matching_percentage:.2f}%")

# Optional: Display some of the matching and non-matching values
print("\nSample of matching values:")
print(list(common_values)[:5])  # Show first 5 matching values

print("\nSample of values in df1 but not in df2:")
print(list(set1 - set2)[:5])  # Show first 5 values unique to df1

print("\nSample of values in df2 but not in df1:")
print(list(set2 - set1)[:5])  # Show first 5 values unique to df2
