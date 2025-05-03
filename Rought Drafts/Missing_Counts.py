import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
import importlib.util
import sys

# Load the dataset
df = pd.read_csv('UCS-Satellite-Database 5-1-2023.csv')

def analyze_all_features():
    """First run SelectKBest on all features"""
    
    # Create a copy of the dataframe
    df_analysis = df.copy()
    
    # Convert categorical features to numeric
    categorical_cols = df_analysis.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_analysis[col] = pd.factorize(df_analysis[col])[0]
    
    # Fill missing values with median
    df_analysis = df_analysis.fillna(df_analysis.median())
    
    # Prepare features and target
    X = df_analysis.drop('Class of Orbit', axis=1)  # All features except target
    y = df_analysis['Class of Orbit']  # Target
    
    # Apply SelectKBest
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P-value': selector.pvalues_
    })
    
    # Sort by score
    feature_scores = feature_scores.sort_values('Score', ascending=False)
    
    # Print all feature importance
    print("\n=== ALL FEATURES IMPORTANCE ANALYSIS ===")
    print("-" * 50)
    print("\nFeature Importance Scores (sorted by importance):")
    for i, (feature, score, p_value) in enumerate(feature_scores.itertuples(index=False)):
        print(f"{i+1}. {feature}:")
        print(f"   • F-Score: {score:.2f}")
        print(f"   • P-value: {p_value:.2e}")
        print(f"   • Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'}")

def analyze_core_features():
    """Then analyze inclination and orbit class specifically"""
    
    # Create a copy with just core features
    df_core = df[['Inclination (degrees)', 'Class of Orbit']].copy()
    
    # Encode orbit class
    df_core['Class of Orbit'] = pd.factorize(df_core['Class of Orbit'])[0]
    
    # Fill missing values with median
    df_core = df_core.fillna(df_core.median())
    
    # Prepare features and target
    X = df_core[['Inclination (degrees)']]  # Feature
    y = df_core['Class of Orbit']  # Target
    
    # Apply SelectKBest
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(X, y)
    
    # Get feature scores
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'Score': selector.scores_,
        'P-value': selector.pvalues_
    })
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_scores['Feature'], feature_scores['Score'], color='skyblue')
    
    # Customize the plot
    plt.title('Feature Importance: Inclination vs Orbit Class', pad=20, fontsize=12)
    plt.xlabel('F-Score', fontsize=10)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        p_value = feature_scores['P-value'].iloc[i]
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'Score: {width:.2f}\nP-value: {p_value:.2e}',
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('core_features_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print core feature importance summary
    print("\n=== CORE FEATURES IMPORTANCE ANALYSIS ===")
    print("-" * 50)
    print("\nFeature Importance Scores:")
    for i, (feature, score, p_value) in enumerate(feature_scores.itertuples(index=False)):
        print(f"{i+1}. {feature}:")
        print(f"   • F-Score: {score:.2f}")
        print(f"   • P-value: {p_value:.2e}")
        print(f"   • Significance: {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'NS'}")
    
    # Create box plot to show relationship
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Class of Orbit', y='Inclination (degrees)', data=df_core)
    plt.title('Inclination Distribution by Orbit Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('inclination_by_orbit_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations have been saved as:")
    print("1. core_features_importance.png - Bar chart showing feature importance score")
    print("2. inclination_by_orbit_class.png - Box plot showing inclination distribution by orbit class")

def get_rmse_and_std_from_ml():
    """Import ML.py and extract RMSE and standard deviation of lifetimes."""
    # Dynamically import ML.py as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location("ml_module", "../ML.py")
    ml = importlib.util.module_from_spec(spec)
    sys.modules["ml_module"] = ml
    spec.loader.exec_module(ml)
    # Try to get rmse and std_lifetime if they exist
    rmse = getattr(ml, 'rmse', None)
    std_lifetime = getattr(ml, 'std_lifetime', None)
    return rmse, std_lifetime

# Get RMSE and standard deviation from ML.py
rmse, std_lifetime = get_rmse_and_std_from_ml()

if rmse is not None and std_lifetime is not None:
    # Plot RMSE vs Standard Deviation
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Standard Deviation\nof Lifetimes', 'Model RMSE'], [std_lifetime, rmse], color=['skyblue', 'lightgreen'])
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f} yrs', ha='center', va='bottom', fontsize=12)
    plt.ylabel('Years')
    plt.title('Comparison: Natural Variation vs Model Error')
    plt.tight_layout()
    plt.savefig('rmse_vs_std_comparison.png', dpi=300)
    plt.show()
else:
    print("Could not import RMSE and standard deviation from ML.py. Make sure they are defined as variables in ML.py.")

# Run both analyses
analyze_all_features()
analyze_core_features() 