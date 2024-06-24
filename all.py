import pandas as pd

# Load the data
df = pd.read_csv('path_to_your_file.csv')

# Assume the columns are named appropriately, or rename them if needed
true_outcomes = df.iloc[:, 1]  # Second column
predicted_outcomes = df.iloc[:, 2]  # Third column

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Calculate metrics
accuracy = accuracy_score(true_outcomes, predicted_outcomes)
recall = recall_score(true_outcomes, predicted_outcomes)
precision = precision_score(true_outcomes, predicted_outcomes)
f1 = f1_score(true_outcomes, predicted_outcomes)
conf_matrix = confusion_matrix(true_outcomes, predicted_outcomes)

# Display metrics
metrics = {
    'Accuracy': accuracy,
    'Recall': recall,
    'Precision': precision,
    'F1 Score': f1,
    'Confusion Matrix': conf_matrix
}

for metric, value in metrics.items():
    if metric != 'Confusion Matrix':
        print(f"{metric}: {value:.2f}")
    else:
        print(f"{metric}:\n{value}")

# Optionally, display the metrics in a DataFrame for better readability
metrics_df = pd.DataFrame([metrics], columns=metrics.keys())
import ace_tools as tools; tools.display_dataframe_to_user(name="Classification Metrics", dataframe=metrics_df)
