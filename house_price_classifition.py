import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set random seed for reproducibility
np.random.seed(42)

# Define column names for the Boston Housing dataset
column_names = [
    'CRIM',     # per capita crime rate by town
    'ZN',       # proportion of residential land zoned for lots over 25,000 sq.ft.
    'INDUS',    # proportion of non-retail business acres per town
    'CHAS',     # Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    'NOX',      # nitric oxides concentration (parts per 10 million)
    'RM',       # average number of rooms per dwelling
    'AGE',      # proportion of owner-occupied units built prior to 1940
    'DIS',      # weighted distances to five Boston employment centres
    'RAD',      # index of accessibility to radial highways
    'TAX',      # full-value property-tax rate per $10,000
    'PTRATIO',  # pupil-teacher ratio by town
    'B',        # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    'LSTAT',    # % lower status of the population
    'MEDV'      # Median value of owner-occupied homes in $1000's (TARGET)
]

try:
    # Load the dataset with proper column names and handle whitespace separation
    df = pd.read_csv('housing.csv', header=None, sep='\s+', names=column_names)
    
    print("\n=== Dataset Information ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    # Visualize the distribution of the target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MEDV'], kde=True)
    plt.title('Distribution of House Prices (MEDV)')
    plt.xlabel('Price in $1000s')
    plt.ylabel('Count')
    plt.savefig('price_distribution.png')
    print("\nSaved price distribution plot as 'price_distribution.png'")
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    print("Saved correlation heatmap as 'correlation_heatmap.png'")
    
    # Split the dataset into features and target variable
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the linear regression model
    model = LinearRegression()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                              cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)
    
    # Train the final model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print model evaluation
    print("\n=== Model Evaluation ===")
    print(f"Cross-Validation RMSE: {cv_rmse_scores.mean():.2f} ± {cv_rmse_scores.std():.2f}")
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test R² Score: {r2:.4f}")
    
    # Plot feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nSaved feature importance plot as 'feature_importance.png'")
    
    # Plot predicted vs actual values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Prices ($1000s)')
    plt.ylabel('Predicted Prices ($1000s)')
    plt.title('Actual vs Predicted House Prices')
    plt.tight_layout()
    plt.savefig('predictions_vs_actual.png')
    print("Saved predictions vs actual plot as 'predictions_vs_actual.png'")
    
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    raise
