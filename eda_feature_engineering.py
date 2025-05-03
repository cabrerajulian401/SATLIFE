import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Load the original dataset
df = pd.read_csv('UCS-Satellite-Database 5-1-2023.csv')

print(df.shape)


irrelevant_cols = [
    'NORAD Number', 'Name of Satellite', 'Alternate Names', 'COSPAR Number',
    'Comments', 'Website', 'Source', 'Operator Contact Info', 'Power (watts)'
   
]
df = df.drop(columns=[col for col in irrelevant_cols if col in df.columns])


target = 'Expected Lifetime (yrs.)'
df[target] = pd.to_numeric(df[target], errors='coerce')
initial_shape = df.shape
df = df.dropna(subset=[target])
print( {initial_shape[0] - df.shape[0]} )


for col in df.select_dtypes(include=[float, int]).columns:
    if col != target:
        df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna('Unknown', inplace=True)


before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
if before != after:
    print(f"Removed {before - after} duplicate rows.")


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature selection with SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)  # Select top 5 features
selector.fit(df_encoded.drop(columns=[target]), df_encoded[target])
selected_features = df_encoded.drop(columns=[target]).columns[selector.get_support()]

# Remove contractor and '_Unknown' features
selected_features = [feat for feat in selected_features if not feat.startswith('Contractor_')]
selected_features = [feat for feat in selected_features if '_Unknown' not in feat]
print("Top features after removing contractor and '_Unknown' columns:", list(selected_features))

