import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# 1. Load and clean data dropping any rows with missing values
df = pd.read_csv("UCS-Satellite-Database 5-1-2023.csv")
df['Expected Lifetime (yrs.)'] = pd.to_numeric(df['Expected Lifetime (yrs.)'], errors='coerce')
df['Inclination (degrees)'] = pd.to_numeric(df['Inclination (degrees)'], errors='coerce')
df = df.dropna(subset=['Expected Lifetime (yrs.)', 'Inclination (degrees)', 'Class of Orbit'])

# Print columns to debug
print(df.columns)


# 2. One-hot encode orbit class into Binary Values
df_encoded = pd.get_dummies(df, columns=['Class of Orbit'])

# 3. Select features and target
features = ['Inclination (degrees)'] + [col for col in df_encoded.columns if col.startswith('Class of Orbit_')]
X = df_encoded[features]
y = df_encoded['Expected Lifetime (yrs.)']

# 4. Scale inclination since it is a Numerical Value, could lead to bias if I do not
scaler = StandardScaler()
X['Inclination (degrees)'] = scaler.fit_transform(X[['Inclination (degrees)']])

# 5. Train/test split- 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 6. Model training with grid search
param_grid = {'fit_intercept': [True, False], 'positive': [True, False]}
grid = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
model = grid.best_estimator_

# 7. Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {grid.best_params_}")
print(f"Test RMSE: {rmse:.3f} years")
print(f"Test MAE: {mae:.3f} years")
print(f"Test R²: {r2:.3f}")

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f} years")
print(f"Test R²: {r2:.3f}")

# Calculate standard deviation of expected lifetimes so we can compare it to our RMSE later on 
std_lifetime = df['Expected Lifetime (yrs.)'].std()

# Plot Actual vs Predicted with metrics in the plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='green', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Lifetime (years)')
plt.ylabel('Predicted Lifetime (years)')
plt.title('Actual vs Predicted Satellite Lifetime')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Adding metrics as text box in the plot for ease of reading 
plt.text(0.05, 0.95, f'RMSE: {rmse:.2f} years\nR²: {r2:.3f}',
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()


