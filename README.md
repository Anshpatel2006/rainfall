# 🌧️ Rainfall Prediction Project

## 📋 Project Overview

This project implements a **Linear Regression model** to predict precipitation levels in Austin, Texas using historical weather data. The goal is to understand the relationships between various weather parameters and precipitation levels, providing insights into weather patterns and prediction capabilities.

## 🎯 Problem Statement

**Objective:** Predict precipitation levels (PrecipitationSumInches) based on weather parameters such as temperature, humidity, dew point, visibility, wind speed, and atmospheric pressure.

**Dataset:** Austin Weather Dataset containing historical weather records from Austin, Texas with 21 weather attributes and 1,319 data points.

## 📊 Dataset Information

- **Source:** Austin Weather Dataset (austin_weather.csv)
- **Records:** 1,319 weather observations
- **Features:** 21 weather parameters
- **Target Variable:** PrecipitationSumInches
- **Time Period:** Historical weather data for Austin, Texas

### Key Features Used:
- Temperature (High, Average, Low)
- Dew Point (High, Average, Low)
- Humidity (High, Average, Low)
- Sea Level Pressure (High, Average, Low)
- Visibility (High, Average, Low)
- Wind Speed (High, Average, Gust)

## 🛠️ Methodology

### 1. Data Preprocessing
- **Data Cleaning:** Handled missing values and special characters ('T' for trace, '-' for missing)
- **Data Type Conversion:** Converted string columns to numeric format
- **Missing Value Imputation:** Used mean imputation for numeric columns
- **Feature Selection:** Selected 18 relevant weather features for modeling

### 2. Exploratory Data Analysis (EDA)
- **Data Visualization:** Created comprehensive plots showing:
  - Precipitation trends over time
  - Distribution of precipitation values
  - Scatter plots of weather parameters vs precipitation
  - Monthly precipitation patterns
- **Correlation Analysis:** Identified relationships between features and target variable

### 3. Model Development
- **Algorithm:** Linear Regression (sklearn.linear_model.LinearRegression)
- **Data Split:** 80% training, 20% testing
- **Feature Scaling:** StandardScaler for normalization
- **Cross-validation:** Random state = 42 for reproducibility

### 4. Model Evaluation
- **Metrics Used:**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²) Score
- **Visualization:** Actual vs Predicted plots, Residuals analysis

## 📈 Key Findings

### Model Performance
- **Training R² Score:** [Value from your model]
- **Test R² Score:** [Value from your model]
- **Test RMSE:** [Value from your model] inches
- **Model explains:** [Percentage]% of variance in precipitation

### Feature Importance (Top 5)
1. **[Feature Name]:** [Coefficient Value]
2. **[Feature Name]:** [Coefficient Value]
3. **[Feature Name]:** [Coefficient Value]
4. **[Feature Name]:** [Coefficient Value]
5. **[Feature Name]:** [Coefficient Value]

### Weather Insights
- **Seasonal Patterns:** [Your findings about seasonal precipitation]
- **Key Correlations:** [Strongest correlations with precipitation]
- **Data Quality:** Successfully handled missing values and special characters

## 📁 Project Structure

```
rainfall-prediction-project/
├── README.md                    # This file
├── rainfall_prediction.ipynb    # Main Jupyter notebook
├── austin_weather.csv          # Dataset
└── 1.ipynb                     # Alternative implementation
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Running the Project
1. **Clone or download** the project files
2. **Ensure** `austin_weather.csv` is in the same directory
3. **Open** `rainfall_prediction.ipynb` in Jupyter Notebook
4. **Run all cells** sequentially

### Alternative Implementation
- Use `1.ipynb` for a simpler implementation approach

## 📊 Visualizations Included

1. **Precipitation Trends Over Time** - Shows temporal patterns
2. **Distribution of Precipitation** - Histogram of precipitation values
3. **Weather Parameter Scatter Plots** - Temperature, Humidity, Wind vs Precipitation
4. **Monthly Precipitation Patterns** - Seasonal analysis
5. **Correlation Matrix** - Feature relationships heatmap
6. **Model Performance Plots** - Actual vs Predicted, Residuals
7. **Feature Importance** - Coefficient analysis

## 🔍 Analysis Process

### Step 1: Data Loading and Exploration
- Loaded Austin weather dataset
- Explored data structure and missing values
- Identified data types and unique values

### Step 2: Data Cleaning
- Converted Date column to datetime
- Handled special characters ('T', '-')
- Converted string columns to numeric
- Imputed missing values with mean

### Step 3: Feature Engineering
- Selected relevant weather features
- Created feature matrix and target vector
- Verified data quality

### Step 4: Exploratory Analysis
- Generated comprehensive visualizations
- Analyzed correlations between features
- Identified seasonal patterns

### Step 5: Model Development
- Split data into training and testing sets
- Scaled features using StandardScaler
- Trained Linear Regression model
- Generated predictions

### Step 6: Model Evaluation
- Calculated performance metrics
- Created visualization plots
- Analyzed feature importance
- Interpreted results

## 📋 Technical Details

### Libraries Used
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computations
- **matplotlib:** Data visualization
- **scikit-learn:** Machine learning algorithms

### Model Specifications
- **Algorithm:** Linear Regression
- **Feature Scaling:** StandardScaler
- **Train-Test Split:** 80-20 ratio
- **Random State:** 42 (for reproducibility)

### Data Processing
- **Missing Value Strategy:** Mean imputation
- **Special Character Handling:** 'T' → 0.01, '-' → 0
- **Feature Selection:** 18 weather parameters
- **Data Types:** All features converted to numeric

## 🎯 Business Applications

This rainfall prediction model can be applied in:

1. **Agriculture:** Crop planning and irrigation scheduling
2. **Urban Planning:** Flood prevention and water management
3. **Transportation:** Road safety and traffic management
4. **Energy:** Hydroelectric power generation planning
5. **Insurance:** Risk assessment for weather-related claims

## 🔮 Future Improvements

### Model Enhancements
- **Advanced Algorithms:** Random Forest, XGBoost, Neural Networks
- **Feature Engineering:** Lag features, rolling averages, seasonal indicators
- **Ensemble Methods:** Combine multiple models for better accuracy

### Data Improvements
- **Additional Features:** Satellite data, atmospheric pressure gradients
- **Temporal Features:** Day of year, season indicators
- **Geographic Features:** Elevation, proximity to water bodies

### Technical Improvements
- **Hyperparameter Tuning:** Grid search, cross-validation
- **Model Interpretability:** SHAP values, feature importance analysis
- **Real-time Prediction:** API development for live weather data

## 📚 References

- **Dataset Source:** Austin Weather Dataset
- **Machine Learning Framework:** scikit-learn
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib

## 👨‍💻 Author

**Student Name**  
**Course:** Module 13 - Rainfall Prediction Project  
**Institution:** [Your Institution]  
**Date:** [Current Date]

## 📄 License

This project is created for educational purposes as part of the Module 13 assignment.

---

**Note:** This README provides a comprehensive overview of the Rainfall Prediction project, including methodology, findings, and technical details. Update the placeholder values (marked with [brackets]) with your actual model results and findings. 