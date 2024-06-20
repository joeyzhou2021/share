import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score, fbeta_score, make_scorer
from rdkit import Chem

# Function to check for the presence of a pyridine ring
def has_pyridine(smiles):
    mol = Chem.MolFromSmiles(smiles)
    pyridine = Chem.MolFromSmarts('c1ccncc1')
    return int(mol.HasSubstructMatch(pyridine))

# Load the data from the pickle file
pkl_file_path = 'path_to_your_pkl_file.pkl'
with open(pkl_file_path, 'rb') as file:
    substances_data = pickle.load(file)

# Convert the substances data to a DataFrame
substances_df = pd.DataFrame(substances_data, columns=['SMILES'])

# Add a column indicating the presence of a pyridine ring
substances_df['has_pyridine'] = substances_df['SMILES'].apply(has_pyridine)

# Load the feature data from the CSV file
features_file_path = 'path_to_features_file.csv'
features_data = pd.read_csv(features_file_path)

# Load the target variable data from the CSV file
target_file_path = 'path_to_target_file.csv'
target_data = pd.read_csv(target_file_path)

# Ensure the target data has only one column and rename it to 'class' for clarity
target_data.columns = ['class']

# Merge the feature data with the substances data to include the pyridine feature
features_data = features_data.join(substances_df['has_pyridine'])

# Separate the features and target variable
X = features_data.drop(columns=['SMILES'])
y = target_data['class']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb = XGBClassifier()

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
}

# Define the scoring function for F_0.5 macro
fbeta_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=fbeta_scorer, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the entire test set
y_pred = best_model.predict(X_test)
print("Performance on the entire test set:")
print("Precision:", precision_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("F0.5 Macro:", fbeta_score(y_test, y_pred, beta=0.5, average='macro'))

# Separate the test set into substances with pyridine and without pyridine
X_test_with_pyridine = X_test[features_data.loc[y_test.index, 'has_pyridine'] == 1]
y_test_with_pyridine = y_test[features_data.loc[y_test.index, 'has_pyridine'] == 1]

X_test_without_pyridine = X_test[features_data.loc[y_test.index, 'has_pyridine'] == 0]
y_test_without_pyridine = y_test[features_data.loc[y_test.index, 'has_pyridine'] == 0]

# Evaluate the model on substances with pyridine
y_pred_with_pyridine = best_model.predict(X_test_with_pyridine)
print("Performance on substances with pyridine:")
print("Precision:", precision_score(y_test_with_pyridine, y_pred_with_pyridine))
print("Balanced Accuracy:", balanced_accuracy_score(y_test_with_pyridine, y_pred_with_pyridine))
print("F0.5 Macro:", fbeta_score(y_test_with_pyridine, y_pred_with_pyridine, beta=0.5, average='macro'))

# Evaluate the model on substances without pyridine
y_pred_without_pyridine = best_model.predict(X_test_without_pyridine)
print("Performance on substances without pyridine:")
print("Precision:", precision_score(y_test_without_pyridine, y_pred_without_pyridine))
print("Balanced Accuracy:", balanced_accuracy_score(y_test_without_pyridine, y_pred_without_pyridine))
print("F0.5 Macro:", fbeta_score(y_test_without_pyridine, y_pred_without_pyridine, beta=0.5, average='macro'))
