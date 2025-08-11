import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('UCS-Satellite-Database 5-1-2023.csv') 

def create_missing_values_chart():
    
    
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data/len(df))*100
    
    
    missing_df = pd.DataFrame({
        'Feature': missing_data.index,
        'Missing Count': missing_data.values,
        'Percentage': missing_percentages.values
    })
    
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=True)
    
    
    plt.figure(figsize=(12, 8))
    
    
    bars = plt.barh(missing_df['Feature'], missing_df['Missing Count'], color='skyblue')
    
   
    plt.title('Missing Values by Feature', pad=20, fontsize=12)
    plt.xlabel('Number of Missing Values', fontsize=10)
    
   
    plt.yticks(range(len(missing_df['Feature'])), missing_df['Feature'], fontsize=5)  # Smaller font size
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        percentage = missing_df['Percentage'].iloc[i]
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{int(width):,} ({percentage:.1f}%)',
                ha='left', va='center', fontsize=5)  # Smaller font size for labels
    
    
    plt.tight_layout()
    
   
    plt.savefig('missing_values_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_missing_data_metrics():

    
    # Basic counts
    total_columns = len(df.columns)
    missing_data = df.isnull().sum()
    columns_with_missing = missing_data[missing_data > 0]
    
    # Calculate key metrics
    column_completeness_ratio = (total_columns - len(columns_with_missing)) / total_columns
    missing_columns_percentage = (len(columns_with_missing) / total_columns) * 100
    
   

    
    # Calculating severity levels
    severe_missing = len(missing_data[missing_data > len(df)*0.5])  # >50% missing
    moderate_missing = len(missing_data[(missing_data > len(df)*0.2) & (missing_data <= len(df)*0.5)])  # 20-50% missing
    mild_missing = len(missing_data[(missing_data > 0) & (missing_data <= len(df)*0.2)])  # <20% missing
    
   
    # Create severity visualization
    plt.figure(figsize=(10, 6))
    severity_data = [severe_missing, moderate_missing, mild_missing, 
                    total_columns - len(columns_with_missing)]
    labels = ['Severe (>50%)', 'Moderate (20-50%)', 'Mild (<20%)', 'Complete']
    colors = ['red', 'orange', 'yellow', 'green']
    
    plt.pie(severity_data, labels=labels, colors=colors, autopct='%1.1f%%')
    plt.title('Column Completeness Distribution')
    plt.axis('equal')
    
    
    plt.savefig('severity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
    for col, count in columns_with_missing.sort_values(ascending=False).items():
        percentage = (count/len(df))*100
        severity = "SEVERE" if percentage > 50 else "MODERATE" if percentage > 20 else "MILD"
        print(f"• {col}: {count:,} missing ({percentage:.1f}%) - {severity}")


create_missing_values_chart()
analyze_missing_data_metrics()
