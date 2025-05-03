import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load and clean data
df = pd.read_csv("UCS-Satellite-Database 5-1-2023.csv")
df['Expected Lifetime (yrs.)'] = pd.to_numeric(df['Expected Lifetime (yrs.)'], errors='coerce')
df['Inclination (degrees)'] = pd.to_numeric(df['Inclination (degrees)'], errors='coerce')
df = df.dropna(subset=['Expected Lifetime (yrs.)', 'Inclination (degrees)', 'Class of Orbit'])

# 1. Distribution of Satellite Lifetimes
plt.figure(figsize=(10, 6))
plt.hist(df['Expected Lifetime (yrs.)'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Expected Lifetime (years)')
plt.ylabel('Number of Satellites')
plt.title('Distribution of Satellite Lifetimes')
plt.tight_layout()
plt.savefig('lifetime_distribution.png', dpi=300)
plt.show()

# 2. Boxplots of Lifetime by Orbit Class
orbit_classes = df['Class of Orbit'].unique()
data = [df[df['Class of Orbit'] == oc]['Expected Lifetime (yrs.)'] for oc in orbit_classes]
plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=orbit_classes)
plt.xlabel('Orbit Class')
plt.ylabel('Expected Lifetime (years)')
plt.title('Lifetime Distribution by Orbit Class')
plt.tight_layout()
plt.savefig('lifetime_by_orbit_class.png', dpi=300)
plt.show()

# 3. Correlation Analysis: Inclination vs Expected Lifetime
r, p = pearsonr(df['Inclination (degrees)'], df['Expected Lifetime (yrs.)'])
plt.figure(figsize=(10, 6))
plt.scatter(df['Inclination (degrees)'], df['Expected Lifetime (yrs.)'], alpha=0.5, color='blue')
z = np.polyfit(df['Inclination (degrees)'], df['Expected Lifetime (yrs.)'], 1)
plt.plot(df['Inclination (degrees)'], np.poly1d(z)(df['Inclination (degrees)']), "r--", label='Trend Line')
plt.xlabel('Inclination (degrees)')
plt.ylabel('Expected Lifetime (years)')
plt.title(f'Inclination vs Lifetime\nCorrelation r = {r:.2f}, p = {p:.2e}')
plt.legend()
plt.tight_layout()
plt.savefig('inclination_vs_lifetime_correlation.png', dpi=300)
plt.show()

print(f"Correlation coefficient (r): {r:.3f}")
print(f"P-value: {p:.3e}")
