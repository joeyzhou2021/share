import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
X_df = pd.read_csv('aromatase_inhibitor_model_X.csv')
y_df = pd.read_csv('aromatase_inhibitor_model_y.csv')

# Merge the dataframes (assuming they have a common identifier column)
df = pd.merge(X_df, y_df, on='identifier')

# List of features to analyze
features_to_analyze = ['ecfp_3651', 'fr_pyridine', 'fp_1234', 'fp_5678']  # Add or remove features as needed

def calculate_metrics(group):
    return pd.Series({
        'accuracy': accuracy_score(group['measured outcome'], group['predicted outcome']),
        'precision': precision_score(group['measured outcome'], group['predicted outcome']),
        'recall': recall_score(group['measured outcome'], group['predicted outcome']),
        'f1': f1_score(group['measured outcome'], group['predicted outcome']),
        'count': len(group)
    })

results = {}

for feature in features_to_analyze:
    # For binary features
    if df[feature].nunique() == 2:
        df[f'{feature}_present'] = df[feature] > 0
    # For continuous features, use median as threshold
    else:
        median = df[feature].median()
        df[f'{feature}_present'] = df[feature] > median
    
    # Calculate performance metrics
    performance_metrics = df.groupby(f'{feature}_present').apply(calculate_metrics)
    results[feature] = performance_metrics
    
    # Visualize performance comparison
    plt.figure(figsize=(10, 6))
    performance_metrics[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar')
    plt.title(f'Model Performance: {feature}')
    plt.ylabel('Score')
    plt.xticks([0, 1], [f'No {feature}', f'{feature} Present'], rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{feature}_performance_comparison.png')
    plt.close()
    
    # Distribution of predicted probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='predicted outcome', hue=f'{feature}_present', element='step', stat='density', common_norm=False)
    plt.title(f'Distribution of Predicted Outcomes: {feature}')
    plt.xlabel('Predicted Outcome')
    plt.savefig(f'{feature}_prediction_distribution.png')
    plt.close()

# Save results
with pd.ExcelWriter('feature_specific_performance_results.xlsx') as writer:
    for feature, metrics in results.items():
        metrics.to_excel(writer, sheet_name=feature)

print("\nAnalysis complete. Results and visualizations have been saved.")

# Print summary of results
for feature, metrics in results.items():
    print(f"\nPerformance metrics for {feature}:")
    print(metrics)
    
    # Calculate and print the difference in metrics
    diff = metrics.iloc[1] - metrics.iloc[0]
    print(f"\nDifference in metrics (Present - Absent):")
    print(diff[['accuracy', 'precision', 'recall', 'f1']])
    print("--------------------")
