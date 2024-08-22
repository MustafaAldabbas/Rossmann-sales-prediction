
# Rossmann Sales Prediction
## Author 
* Mustafa Aldabbas, connect with me on [LinkedIn](https://www.linkedin.com/in/mustafa-aldabbas-85256b95/), [on X](https://x.com/Mustafa_dabbas)
### **Bio**
As a full-stack Data Analyst, I specialize in comprehensive data solutions from coding in Python and SQL to advanced predictive modeling and interactive visualizations with Streamlit and Tableau. My experience spans detailed exploratory data analysis to deploying predictive models for retail forecasts, notably enhancing forecasting capabilities for major brands like Rossmann. I blend technical expertise with strategic insights to drive significant business outcomes in competitive markets.

# [Presentation](https://docs.google.com/presentation/d/1YLVU45Rn9iQhw_2XDuCX_EqTijWmMwxz/edit?usp=sharing&ouid=118224995700991179223&rtpof=true&sd=true)

#Streamlit App Demo
Here is a recorded demo of the Streamlit app:
[Watch the demo video](https://drive.google.com/file/d/1lErNDzd0F0_yixbEye-LMcozI6eHNqbS/view?usp=sharing)

## Project Overview
This project aims to predict daily sales for Rossmann stores using historical sales data, store characteristics, promotions, and other relevant factors. The goal is to identify key factors influencing sales, build an accurate predictive model, and provide actionable insights to optimize sales strategies in a competitive retail environment.

## Objectives
- **Predict Daily Sales:** Develop models to accurately forecast daily sales for Rossmann stores.
- **Understand Sales Influences:** Analyze how various factors like promotions, store types, and competition affect sales.
- **Provide Strategic Insights:** Identify actionable recommendations for enhancing sales performance based on data-driven insights.

## Datasets
![Sales Distribution](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/streamlit%20pics%20/Project%20Objectibes%20.pptx.png)

- **train.csv:** Historical sales data including daily sales figures for Rossmann stores.
- **store.csv:** Store-specific features such as store type, assortment type, and competition details.
- **test.csv:** Data used for predicting sales without the actual sales figures.


## Project Structure

1. **Data Loading, Cleaning, and Preprocessing**
    - **Loading:** Imported datasets into Pandas DataFrames.
    - **Merging:** Combined `train.csv` and `store.csv` on the `Store` column.
    - **Preprocessing:** Handled missing values, converted data types, and removed outliers.
 

2. ## **Exploratory Data Analysis (EDA)**
    - ## **Univariate Analysis:**
      
    ![Sales Distribution](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/UNI%20variante%20/Sales%20distribution%20.png)

    - ## **Bivariate Analysis:**
      
    ![Sales and promotion](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/EDA/Sales%20and%20Promotion.png)

    - ## **Sales over time:**
      
    ![sales Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/EDA/sales%20by%20month%20.png)

    - ## **Sales, Customer, Competition:**
      
    ![sales Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/EDA/sales%20cluster.png)
    
    


3. ## **Feature Engineering**
   
   ![Gross Income Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/streamlit%20pics%20/Feature%20engineering.pptx.png)
    - **Binary Features:** Created binary indicators for promotions and competition effects.
    - **Lag Features:** Added lagged sales data to capture time-dependent patterns.
    - **Moving Averages:** Calculated moving averages for sales to smooth out fluctuations.
    - **Encoding:** Applied label encoding and one-hot encoding to categorical variables.

5. ## **Model Building**
   ![Gross Income Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/streamlit%20pics%20/Modeling.pptx.png)
    
    - **Random Forest Regressor:** Trained an initial model to establish a baseline.
    - **LightGBM Regressor:** Developed a LightGBM model for comparison.
    - **XGBoost:** Built and tuned an XGBoost model, which showed superior performance.

6. ## **Model Evaluation and Selection**
   ![Gross Income Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/streamlit%20pics%20/model%20evaluation.png)
    
    - **Metrics:** Evaluated models using RMSE, MAE, and R².
    - **Hyperparameter Tuning:** Optimized model performance through grid search and cross-validation.
    - **Model Comparison:** Compared different models and selected the best one for deployment.

7. ## **Predictions and Deployment**
   
   ![Gross Income Over Time](https://github.com/MustafaAldabbas/Rossmann-sales-prediction/blob/main/Visuals/streamlit%20pics%20/2222%20predictions.png)
    - **Final Model Training:** Retrained the best model on the full training set.
    - **Prediction:** Generated sales predictions on the test dataset.
    - **Deployment:** Saved the model for future use and prepared a submission file.

# Results
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
