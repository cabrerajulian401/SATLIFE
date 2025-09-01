# SATLIFE: Advancing Satellite Lifetime Prediction through Machine Learning

## Overview
SATLIFE addresses the critical challenge of accurately predicting the operational lifetime of Earth-orbiting satellites using machine learning techniques. This project develops a robust, data-driven predictive model capable of providing reliable estimates of satellite lifespan in years, leveraging comprehensive satellite operational data and advanced regression modeling.

## Project Objectives
- Develop and evaluate a machine learning regression model for satellite lifetime prediction
- Analyze factors influencing satellite operational longevity
- Provide insights for strategic planning in satellite design and space debris mitigation
- Enable optimized resource allocation within complex space programs

## Dataset
The project utilizes the Union of Concerned Scientists (UCS) Satellite Database (May 1, 2023 snapshot), containing:

- 7,562 distinct satellites
- 67 features encompassing operational parameters, physical characteristics, and orbital specifics
- Key attributes include power consumption, mass, orbital class, inclination, launch information, and operator details

## Key Findings

### Strong Predictive Performance
- **Root Mean Squared Error (RMSE)**: 2.010 years
- **R² Score**: 0.662 (explaining 66.2% of variance in satellite lifetimes)
- **Mean Absolute Error (MAE)**: ~1.926 years

### Critical Insights
- **Orbital Inclination Impact**: Strong negative correlation (-0.517) between satellite inclination and expected lifetime
- **Mission Type Variation**: Significant differences in operational lifetimes across orbit classes (LEO, GEO, MEO, Elliptical)
- **Feature Selection**: Analysis focused on **two primary feature types**:
  - **Inclination (degrees)**: Numerical feature representing orbital inclination angle
  - **Class of Orbit**: One-hot encoded categorical features for different orbital classes (GEO, LEO, MEO, Elliptical, etc.)

## Methodology

### Data Preprocessing
- **Missing Data Handling**: Strategic removal of rows with missing values in critical features (Expected Lifetime, Inclination, Class of Orbit)
- **Feature Engineering**: One-hot encoding for categorical variables, particularly orbital class
- **Feature Selection**: Targeted selection focusing on **inclination and orbital class features only**
- **Normalization**: StandardScaler applied to numerical features (Inclination)
- **Train-Test Split**: 80% training, 20% testing with reproducible random state

### Model Architecture
Multiple Linear Regression (MLR) selected for:

- Interpretability and transparency
- Computational efficiency
- Strong baseline performance for linear relationships
- Clear coefficient interpretation for stakeholder communication

**Model Equation:**
```
Predicted_Lifetime = β₀ + β₁(Inclination_scaled) + β₂(Class_of_Orbit_GEO) + ... + βₙ(Class_of_Orbit_Other) + ε
```

**Final Feature Set:**
- 1 numerical feature: `Inclination (degrees)` (standardized)
- Multiple binary features: All one-hot encoded orbital classes (`Class of Orbit_*`)

### Hyperparameter Optimization
- GridSearchCV with 5-fold cross-validation
- Optimized parameters: fit_intercept and positive constraints
- Negative RMSE scoring for performance maximization

## Technical Implementation

### Core Technologies
- **Python**: Primary development language
- **scikit-learn**: Machine learning framework
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **matplotlib**: Data visualization

### Key Scripts
- **ML.py**: Main model training and evaluation
- **Final_code.py**: Complete analysis pipeline with feature selection
- **eda.py**: Exploratory data analysis
- **missing_counts.py**: Missing data analysis
- **eda_feature_engineering.py**: Feature preprocessing pipeline

## Results and Performance

### Model Accuracy
The final model demonstrates practically useful accuracy with predictions typically within ±2 years of actual satellite lifetimes. This level of precision is valuable for:

- Mission planning and resource estimation
- Operational decision-making
- Space debris mitigation strategies

### Model Robustness
Strong consistency between cross-validated training results and test set performance indicates robust generalization capabilities for real-world applications.

## Data Challenges and Limitations

### Pervasive Missing Data
- **>50% missing values**: 43 columns severely constrained feature selection
- **20-50% missing values**: 24 columns required careful imputation strategies
- **Trade-off**: Between sample size and data completeness

### Inherent Limitations
- **Linear Assumptions**: MLR may not capture complex non-linear interactions
- **Data Granularity**: Potential inconsistencies across different eras and operators
- **Model Simplicity**: Does not account for time-dependent effects or complex feature interactions
- **Limited Feature Set**: Final model uses only 2 feature types due to data quality constraints

## Future Enhancements

### Advanced Modeling Approaches
- **Non-linear Algorithms**: Random Forest, Gradient Boosting, Support Vector Regression
- **Deep Learning**: Neural networks for complex pattern recognition
- **Time-Series Analysis**: Incorporating temporal degradation patterns
- **Uncertainty Quantification**: Prediction intervals for risk assessment

### Data Expansion
- Integration of additional data sources (engineering specifications, environmental data)
- Historical operational data for time-series modeling
- Real-time telemetry integration for dynamic predictions

### Causal Analysis
- Exploration of causal inference techniques to understand design choice impacts
- Feature interaction analysis for optimization insights

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/cabrerajulian401/SATLIFE.git
cd SATLIFE

# Install required dependencies
pip install -r requirements.txt

# Run the main analysis
python ML.py

# Run complete analysis pipeline
python Final_code.py

# Perform exploratory data analysis
python eda.py
```

## Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the model's accuracy and expand its capabilities.

## References
- Union of Concerned Scientists. (2023). UCS Satellite Database. Retrieved from [UCS Database](https://www.ucsusa.org/resources/satellite-database)
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
- Harris, C. R., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.

## License
This project is available under the MIT License. See LICENSE file for details.

---

**Note**: This project demonstrates the application of machine learning techniques to aerospace engineering challenges, providing practical insights for satellite operations and space asset management.
