
# Rossmann Sales Prediction
## Author 
* Mustafa Aldabbas, connect with me on [LinkedIn](https://www.linkedin.com/in/mustafa-aldabbas-85256b95/), [on X](https://x.com/Mustafa_dabbas)


# [Presentation](https://docs.google.com/presentation/d/1YLVU45Rn9iQhw_2XDuCX_EqTijWmMwxz/edit?usp=sharing&ouid=118224995700991179223&rtpof=true&sd=true)

## Project Overview
This project aims to predict daily sales for Rossmann stores using historical sales data, store characteristics, promotions, and other relevant factors. The goal is to identify key factors influencing sales, build an accurate predictive model, and provide actionable insights to optimize sales strategies in a competitive retail environment.

## Objectives
- **Predict Daily Sales:** Develop models to accurately forecast daily sales for Rossmann stores.
- **Understand Sales Influences:** Analyze how various factors like promotions, store types, and competition affect sales.
- **Provide Strategic Insights:** Identify actionable recommendations for enhancing sales performance based on data-driven insights.

## Datasets
- **train.csv:** Historical sales data including daily sales figures for Rossmann stores.
- **store.csv:** Store-specific features such as store type, assortment type, and competition details.
- **test.csv:** Data used for predicting sales without the actual sales figures.


## Project Structure

1. **Data Loading, Cleaning, and Preprocessing**
    - **Loading:** Imported datasets into Pandas DataFrames.
    - **Merging:** Combined `train.csv` and `store.csv` on the `Store` column.
    - **Preprocessing:** Handled missing values, converted data types, and removed outliers.
    - **Time Series Analysis:** Conducted analysis to identify trends, seasonality, and cyclical patterns in sales data.

2. **Exploratory Data Analysis (EDA)**
    - **Univariate Analysis:** Examined the distribution of key variables like sales and customer counts.
    ![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Bivariate Analysis:** Explored relationships between variables, such as promotions and sales, using visualizations.
    ![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Correlation Analysis:** Identified relationships between numerical features and the target variable (Sales).
    ![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)


3. **Feature Engineering**
![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Binary Features:** Created binary indicators for promotions and competition effects.
    - **Lag Features:** Added lagged sales data to capture time-dependent patterns.
    - **Moving Averages:** Calculated moving averages for sales to smooth out fluctuations.
    - **Encoding:** Applied label encoding and one-hot encoding to categorical variables.

4. **Model Building**
![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Random Forest Regressor:** Trained an initial model to establish a baseline.
    - **LightGBM Regressor:** Developed a LightGBM model for comparison.
    - **XGBoost:** Built and tuned an XGBoost model, which showed superior performance.

5. **Model Evaluation and Selection**
![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Metrics:** Evaluated models using RMSE, MAE, and R².
    - **Hyperparameter Tuning:** Optimized model performance through grid search and cross-validation.
    - **Model Comparison:** Compared different models and selected the best one for deployment.

6. **Predictions and Deployment**
![Gross Income Over Time](https://github.com/MustafaAldabbas/Machine_learning_superstore/blob/main/my%20pic/gross_income__over_time.png)

    - **Final Model Training:** Retrained the best model on the full training set.
    - **Prediction:** Generated sales predictions on the test dataset.
    - **Deployment:** Saved the model for future use and prepared a submission file.

## Results
- **Best Model:** XGBoost, which provided the lowest RMSE and highest R², was selected as the final model.
- **Insights:** The analysis revealed the significant impact of promotions, store types, and competition on sales.
- **Recommendations:** Suggested strategic initiatives for Rossmann to enhance sales based on the model's predictions.

## How to Use

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/MustafaAldabbas/Rossmann-sales-prediction
    ```
2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Notebook:**
    ```bash
    jupyter notebook Rossmann_Sales_Prediction.ipynb
    ```
4. **Launch Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## Conclusion
- **Key Sales Influencers:** Promotions, store types, and competition significantly impact sales.

- **Predictive Modeling:** Developed accurate sales forecasting models like Random Forest and XGBoost.

- **Promotion Effectiveness:** Targeted promotions effectively increase sales.

- **Tailored Strategies:** Custom strategies for different store types enhance performance.

- **Competitive Insights:** Competitor analysis informs strategic store placement and tactics.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For further inquiries, please contact Mustafa Aldabbas at [Mustafa.aldabbas@outlook.com](mailto:Mustafa.aldabbas@outlook.com).

## How to Run the Streamlit App

1. Clone the repository.
2. Install the required libraries from `requirements.txt`.
3. Run the app using the command:
   ```bash
   streamlit run app.py
   ```