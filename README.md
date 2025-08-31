# 🏠 House Price Prediction with Linear Regression

This project applies **Linear Regression** on the **Boston Housing Dataset** to predict median house prices based on socioeconomic and environmental features. It includes **EDA (exploratory data analysis)**, **data preprocessing**, **cross-validation**, and **model evaluation**.

---

## 📂 Project Structure

├── house_price_classifition.py # Main script (EDA, training, evaluation)
├── housing.csv # Dataset (Boston Housing Dataset)
├── requirements.txt # Dependencies
├── price_distribution.png # Distribution of house prices
├── correlation_heatmap.png # Correlation heatmap of features
├── feature_importance.png # Feature importance from regression coefficients
├── predictions_vs_actual.png # Actual vs predicted values
└── README.md # Project documentation


---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
🚀 How to Run
Clone the repository:


git clone https://github.com/omarfoud/house-price-prediction.git
cd house-price-prediction

Run the script:

python house_price_classifition.py

Outputs:

Console prints:

Dataset info and stats

Cross-validation RMSE

Test RMSE, MAE, and R² score

Saved plots:

price_distribution.png

correlation_heatmap.png

feature_importance.png

predictions_vs_actual.png

🔬 Approach

Dataset: Boston Housing Dataset (13 features + target MEDV).

EDA:

Price distribution visualization.

Feature correlation heatmap.

Preprocessing:

Train-test split (80/20).

Feature scaling using StandardScaler.

Model: Linear Regression.

Evaluation:

Cross-validation RMSE.

Test RMSE, MAE, and R² score.

Visualization:

Feature importance.

Predictions vs actual prices.

📊 Example Output

=== Model Evaluation ===
Cross-Validation RMSE: 4.75 ± 1.20
Test RMSE: 4.92
Test MAE: 3.33
Test R² Score: 0.71

🤝 Contribution
Contributions are welcome! Open an issue or submit a pull request to improve the project.