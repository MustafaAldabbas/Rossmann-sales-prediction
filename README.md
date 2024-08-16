# Rossmann-sales-prediction

## Project Overview
This project aims to predict daily sales for Rossmann stores using historical sales data, store characteristics, promotions, and other relevant factors. The goal is to understand the key factors that influence sales, build a predictive model for accurate sales forecasting, and provide actionable insights for improving sales.

## Objectives
- Predict daily sales for Rossmann stores based on historical data.
- Understand factors that influence sales in retail stores.
- Build a predictive model to forecast sales accurately.
- Identify actionable insights for improving sales.

## Datasets
- **train.csv**: Historical sales data.
- **store.csv**: Store-related features.
- **test.csv**: Data for making predictions.


## Project Structure
1. **Data Loading, Cleaning, and Preprocessing**
   - Loaded datasets into pandas DataFrames.
   - Merged `train.csv` and `store.csv` on the `Store` column.
   - Checked for missing values and handled them with appropriate imputation methods.
   - Converted date columns to datetime format and removed outliers.

2. **Exploratory Data Analysis (EDA)**
   - Conducted univariate analysis to understand the distribution of key variables.
   - Performed bivariate and multivariate analysis to explore relationships between variables.
   - Conducted correlation analysis to identify relationships between numerical features and target variables.

3. **Feature Engineering**
   - Created binary features for specific conditions (e.g., promotions).
   - Engineered features related to competition and promotions.
   - Added lag features and moving averages for sales and customer counts.

4. **Model Building**
   - Built and trained a Random Forest model.
   - Visualized feature importance from the Random Forest model.
   - Developed and trained a LightGBM Regressor model for comparison.

5. **Model Evaluation and Selection**
   - Evaluated models using RMSE, MAE, and R² metrics.
   - Performed hyperparameter tuning to optimize the best model.
   - Compared the performance of all models to select the best one.

6. **Predictions and Deployment**
   - Made predictions on the test dataset using the selected model.
   - Re-trained the selected model on the entire training dataset to make final predictions.
   - Saved the final model and prepared the submission file.

## Results
- The best-performing model was selected based on evaluation metrics.
- Predictions were made on the test dataset, and a final submission file was created.
- Key insights were derived from the analysis, providing actionable recommendations for Rossmann stores.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/MustafaAldabbas/Rossmann-sales-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook to reproduce the results:
   ```bash
   jupyter notebook Rossmann_Sales_Prediction.ipynb
   ```

## Conclusion
This project successfully developed a predictive model for daily sales in Rossmann stores, providing valuable insights into factors that influence sales. The model can be used for future sales forecasting and strategy optimization.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions, please contact [yourname] at [Mustafa.aldabbas@outlook.com].
