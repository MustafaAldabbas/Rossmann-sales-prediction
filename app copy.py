import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from PIL import Image
import io
from dotenv import load_dotenv
import os
import openai
import base64



from functions import (
    load_datasets,
    clean_and_merge_datasets,
    timeseries_eda,
    train_and_evaluate_rf,
    train_and_evaluate_xgboost,
    tune_and_evaluate_xgboost,
    save_and_train_full_model,
    predict_sales_for_test_df,
    plot_sales_comparison_streamlit,
    bivariate_eda,
    perform_store_clustering,
    univariate_edast,
    plot_sales_by_store_type_and_assortment,
    plot_sales_by_school_holiday,
    plot_sales_by_competition_distance,
    plot_sales_by_promo,
    feature_engineering,
)
load_dotenv()
@st.cache_data
def load_datasets():
    # Load datasets from CSV files based on the paths you provided
    train_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/train.csv')
    test_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/test.csv')
    store_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/store.csv')
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'store_df': store_df
    }

def describe_image(image):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)

    # Set the API key
    openai.api_key = os.getenv("api_key")

    # Convert image to base64
    base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message['content']

# Load datasets using the `load_datasets` function
data = load_datasets('config.yaml')
train_df = data['train_df']
store_df = data['store_df']
test_df = data['test_df']

# Clean and merge the datasets
train_df_merged, test_df_merged = clean_and_merge_datasets(train_df, test_df, store_df)

# **Load the pre-processed test dataframe (processed_test_df.pkl)**
test_df_merged = pd.read_pickle("processed_test_df.pkl")

# Set up the Streamlit app
st.set_page_config(page_title="Rossmann Sales Forecasting", layout="wide")

# Sidebar Navigation
st.sidebar.title("Rossmann Sales Forecasting App")
pages = st.sidebar.radio(
    "Navigate to",
    ["Introduction", "EDA", "EDA 2", "Feature Engineering", "Modeling", "Prediction & Visualization", "Conclusion"]
)

# Authors section in sidebar
st.sidebar.markdown("### Authors")
st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/Personal pic/Mustafa HS2.jpg', width=100)
st.sidebar.markdown("Mustafa Aldabbas")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-linkedin-id/)")

# Introduction Page
if pages == "Introduction":
    st.markdown("<h1 style='color: LightBlue;'>Rossmann Sales Forecasting 📈</h1>", unsafe_allow_html=True)
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /rossman.png', width=1000)  # Update this path

    # Load the datasets
    train_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/train.csv')
    test_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/test.csv')
    store_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/store.csv')
    
    st.markdown("### Raw Datasets Preview")
    
    # Create tabs for each dataset
    tab1, tab2, tab3 = st.tabs(["Train Dataset", "Test Dataset", "Store Dataset"])
    
    with tab1:
        st.markdown("**Train Dataset (First 5 rows):**")
        st.dataframe(train_data.head())
    
    with tab2:
        st.markdown("**Test Dataset (First 5 rows):**")
        st.dataframe(test_data.head())
    
    with tab3:
        st.markdown("**Store Dataset (First 5 rows):**")
        st.dataframe(store_data.head())
    
    st.markdown("""
    ## Welcome to the Rossmann Sales Forecasting Project

    The project aims to address a critical business challenge: predicting daily sales for Rossmann stores. The ability to accurately forecast sales is essential for inventory management, staffing, and promotional planning. By leveraging historical sales data and various store-related features, the project seeks to build a robust predictive model that can help Rossmann improve its operational efficiency and optimize sales.

    ### **1. Project Definition and Objectives**

    **Primary Objective:** 
    - Predict daily sales for Rossmann stores using historical sales data and store-specific features.

    **Secondary Objectives:**
    - Understand the key factors influencing sales in retail stores.
    - Develop a predictive model capable of accurately forecasting sales.
    - Identify actionable insights to enhance sales performance based on model predictions.

    **Project Hypothesis or Analytical Questions**
    The project seeks to explore the following hypotheses or questions:

    - **Key Factors:** What are the most significant factors affecting daily sales in Rossmann stores?
    - **Model Accuracy:** Can we develop a model that accurately predicts sales across various store types and conditions?
    - **Promotions Impact:** How do promotions and competitive factors influence store sales?
    - **Seasonality:** How do seasonal trends affect sales, and can they be effectively modeled?

    ### **2. Datasets Summary**

    **store.csv:**
    Contains information about the stores, including store type, assortment type, competition distance, and promotional details.
    - **Total Records:** 1,115
    - **Key Columns:**
      - Store: Store identifier.
      - StoreType: Type of store.
      - Assortment: Type of assortment.
      - CompetitionDistance: Distance to nearest competitor.
      - Promo2: Indicator of ongoing promotion.
    - **Missing Values:** Some missing values in columns like CompetitionDistance, CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2SinceWeek, Promo2SinceYear, and PromoInterval.

    **train.csv:** 
    Contains historical sales data for training the model.
    - **Total Records:** 1,017,209
    - **Key Columns:**
      - Store: Store identifier.
      - DayOfWeek: Day of the week.
      - Date: Date of the transaction.
      - Sales: Sales amount.
      - Customers: Number of customers.
      - Open: Whether the store was open or closed.
      - Promo: Whether a promotion was running.
    - **Missing Values:** None significant.

    **test.csv:** 
    Contains data to be used for making predictions.
    - **Total Records:** 41,088
    - **Key Columns:**
      - Similar to train.csv but without the Sales column since this is the target variable to predict.
    - **Missing Values:** None significant.

    Navigate through the app to explore the data, understand the models, and see the predictions in action.
    """)


# EDA Page
elif pages == "EDA":
    st.title("Exploratory Data Analysis (EDA) 📊")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Univariate Analysis", 
        "Bivariate Analysis", 
        "Time Series Analysis", 
        "Clustering Analysis",
        "Sales by Store Type and Assortment",
    ])

    with tab1:
        st.subheader("Univariate Analysis")
        fig = univariate_edast(train_df_merged)
        st.pyplot(fig)
        
        try:
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            image = Image.open(img_buffer)
            description = describe_image(image)
            st.write("AI-Generated Description:")
            st.write(description)
        except Exception as e:
            st.write(f"An error occurred while generating the image description: {str(e)}")

    with tab2:
        st.subheader("Bivariate Analysis")
        st.markdown("### Sales Distribution")
        bivariate_eda(train_df_merged) 

    with tab3:
        st.subheader("Time Series Analysis")
        timeseries_eda(train_df_merged)

    with tab4:
        st.subheader("Clustering Analysis")
        perform_store_clustering(train_df_merged)

    with tab5:
        st.subheader("Sales by Store Type and Assortment")
        plot_sales_by_store_type_and_assortment(train_df_merged)  # Display the plots

elif pages == "EDA 2":
    st.title("Exploratory Data Analysis (EDA) 📊")
    tab6, tab7, tab8 = st.tabs([
        "Sales by School Holiday",
        "Sales by Competition Distance",
        "Sales by Promotion" 
    ])

    with tab6:
        st.subheader("Sales by School Holiday")
        plot_sales_by_school_holiday(train_df_merged)  # Display the sales by school holiday plot

    with tab7:
        st.subheader("Sales by Competition Distance")
        plot_sales_by_competition_distance(train_df_merged)  # Display the sales by competition distance plot

    with tab8:
        st.subheader("Sales by Promotion")
        plot_sales_by_promo(train_df_merged)  # Display the sales by promotion plot


elif pages == "Feature Engineering":
    st.title("Feature Engineering 🔧")

    st.markdown("### Applying Feature Engineering")
    test_df_merged, train_df_merged, X_train, X_test, y_train, y_test = feature_engineering(train_df_merged, test_df_merged)

    st.markdown("### Feature Engineering Completed")

    st.write("train_df_merged sample:")
    st.write(train_df_merged.head())

    st.write("test_df_merged sample:")
    st.write(test_df_merged.head())

    st.write("X_train sample:")
    st.write(X_train.head())

    st.write("y_train sample:")
    st.write(y_train.head())

elif pages == "Modeling":
    st.title("Modeling 🧠")

    if 'X_train' not in locals() or 'X_test' not in locals():
        st.markdown("Feature engineering and data splitting are being applied...")
        test_df_merged, train_df_merged, X_train, X_test, y_train, y_test = feature_engineering(train_df_merged, test_df_merged)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Train Random Forest", 
        "Train XGBoost", 
        "Tune XGBoost", 
        "Model Performance",
        "Train Full Model"
    ])

    # Your code for each tab goes here...

    with tab1:
        st.subheader("Train Random Forest")
        st.markdown("### Training Random Forest Model")

        rf_model = train_and_evaluate_rf(X_train, y_train, X_test, y_test)
        if rf_model:
            st.write(f"Random Forest Model Performance Metrics:")
            st.write(f"RMSE: {rf_model['metrics']['rmse']}")
            st.write(f"MAE: {rf_model['metrics']['mae']}")
            st.write(f"R2: {rf_model['metrics']['r2']}")
        else:
            st.error("Random Forest model training failed. Please check the logs for more details.")

    with tab2:
        st.subheader("Train XGBoost")
        st.markdown("### Training XGBoost Model")

        xgb_model = train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)
        if xgb_model:
            st.write(f"XGBoost Model Performance Metrics:")
            st.write(f"RMSE: {xgb_model['metrics']['rmse']}")
            st.write(f"MAE: {xgb_model['metrics']['mae']}")
            st.write(f"R2: {xgb_model['metrics']['r2']}")
        else:
            st.error("XGBoost model training failed. Please check the logs for more details.")

    with tab3:
        st.subheader("Tune XGBoost")
        st.markdown("### Hyperparameter Tuning for XGBoost")

        best_xgb_model, test_rmse_best_xgb, test_mae_best_xgb, test_r2_best_xgb, grid_search = tune_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

        st.write("Best Parameters Found:", grid_search.best_params_)
        st.write("Best RMSE:", test_rmse_best_xgb)
        st.write("Best MAE:", test_mae_best_xgb)
        st.write("Best R-squared:", test_r2_best_xgb)

        # Save the best model from hyperparameter tuning
        tuned_model_path = 'best_xgb_model_tuned.pkl'
        joblib.dump(best_xgb_model, tuned_model_path)
        st.write(f"Tuned model saved to {tuned_model_path}")

    with tab4:
        st.subheader("Model Performance")
        st.markdown("### Evaluate Model Performance")

        # Assuming you have stored performance metrics or have a final model to evaluate
        # Display metrics for Random Forest
        if 'rf_model' in locals() and rf_model:
            st.write("Random Forest Model Performance Metrics:")
            st.write(f"RMSE: {rf_model['metrics']['rmse']}")
            st.write(f"MAE: {rf_model['metrics']['mae']}")
            st.write(f"R2: {rf_model['metrics']['r2']}")
        else:
            st.warning("Random Forest model performance is unavailable. Train the model first.")

        # Display metrics for XGBoost
        if 'xgb_model' in locals() and xgb_model:
            st.write("XGBoost Model Performance Metrics:")
            st.write(f"RMSE: {xgb_model['metrics']['rmse']}")
            st.write(f"MAE: {xgb_model['metrics']['mae']}")
            st.write(f"R2: {xgb_model['metrics']['r2']}")
        else:
            st.warning("XGBoost model performance is unavailable. Train the model first.")

    with tab5:
        st.subheader("Train Full Model")
        st.markdown("### Train Model on Full Dataset")

        if st.button("Train Full Model"):
            best_xgb_model, best_xgb_model_full = save_and_train_full_model(best_xgb_model, train_df_merged)
            st.write(f"Tuned model saved to 'best_xgb_model_tuned.pkl'")
            st.write(f"Full trained model saved to 'best_xgb_model_full_trained.pkl'")

# Prediction and Visualization Page
elif pages == "Prediction & Visualization":
    st.title("Prediction & Visualization 🔮")

    tab1, tab2 = st.tabs(["Make Predictions", "Visualize Predictions"])

    with tab1:
        st.subheader("Make Predictions")
        st.markdown("### Load and Use the Trained Model")

        model_path = 'best_xgb_model_full_trained.pkl'
        try:
            loaded_model = joblib.load(model_path)
            st.write(f"Model loaded from {model_path}")

        
            Test_df_future_sale = predict_sales_for_test_df(test_df_merged, model_path)
            
            # Display some predictions
            st.write("Here are some predictions:")
            st.write(Test_df_future_sale[['Date', 'Predicted_Sales']])

        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}. Please train and save a model first.")
        except KeyError as e:
            st.error(f"Error: {e}")
            st.write("Available columns in test_df_merged:", test_df_merged.columns.tolist())
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    with tab2:
        st.subheader("Visualize Predictions")
        st.markdown("### Predicted vs The last two months of the sales history")
        st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Prediction.png', width=1000)  # Update this path

# Conclusion Page
elif pages == "Conclusion":
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /conclusion.png', width=1000)  # Update this path

    st.title("Conclusion 🏁")

    st.markdown("""
    In this project, we set out to achieve three main objectives: understanding the key factors influencing sales in retail stores, developing a predictive model capable of accurately forecasting sales, and identifying actionable insights to enhance sales performance.

    ### **Summary of Objectives Achieved:**

    **1. Understanding the Key Factors Influencing Sales in Retail Stores:**

    Through exploratory data analysis (EDA), we identified several key factors that significantly impact sales in Rossmann stores. These include:

    - **Promotions:** Promotional periods were found to have a substantial positive effect on sales, indicating the importance of strategic promotional planning.

    - **Store Type and Assortment:** Different store types and assortment strategies showed varying sales patterns, suggesting that tailored strategies for different store types could optimize performance.

    - **Competition:** Proximity to competitors and the duration of their presence were also key factors, highlighting the need for competitive analysis in sales forecasting.

    By analyzing correlations and feature importance from models like Random Forest and XGBoost, we were able to pinpoint the variables that most influence sales. This understanding provides a strong foundation for building effective sales strategies.

    **2. Developing a Predictive Model Capable of Accurately Forecasting Sales:**

    We successfully developed and fine-tuned several predictive models, including Random Forest, XGBoost, and LightGBM, to forecast daily sales. The models were evaluated using key metrics such as RMSE and R², ensuring that the predictions are accurate and reliable. Among these models, [mention the best-performing model] showed the highest accuracy, making it a strong candidate for deployment in a real-world scenario.

    **3. Identifying Actionable Insights to Enhance Sales Performance:**

    - **Promotional Impact:** The strong positive correlation between promotions and sales suggests that Rossmann can drive significant sales increases through well-timed and targeted promotions.

    - **Store-Specific Strategies:** The variations in sales patterns across different store types indicate that Rossmann could benefit from customizing sales strategies based on store characteristics.

    - **Competitor Analysis:** Understanding the competitive landscape and its impact on sales can help Rossmann make informed decisions about store locations and competitive positioning.

    ### **Key Takeaways**
    - **Accurate Sales Predictions:** The model provides reliable predictions to assist Rossmann in sales planning.
    - **Actionable Insights:** The analysis identified key factors influencing sales, providing valuable insights for business decisions.

    ### **Next Steps**
    Consider using these predictions to optimize inventory, staffing, and promotional strategies across stores.
    """)
