import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from functions import (
    load_datasets,
    clean_and_merge_datasets,
    timeseries_eda,
    train_and_evaluate_rf,
    feature_engineering,
    tune_and_evaluate_xgboost,
    bivariate_eda,
    perform_store_clustering,
    univariate_eda,
    plot_sales_by_store_type_and_assortment,
    plot_sales_by_school_holiday,
    plot_sales_by_competition_distance,
    plot_sales_by_promo,
    save_and_train_full_model,
    predict_sales_for_test_df,
    plot_sales_comparison_streamlit
)

# Load datasets using the `load_datasets` function
data = load_datasets('config.yaml')
train_df = data['train_df']
store_df = data['store_df']
test_df = data['test_df']

# Clean and merge the datasets
train_df_merged, test_df_merged = clean_and_merge_datasets(train_df, test_df, store_df)

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
    st.markdown("""
    ## Welcome to the Rossmann Sales Forecasting Project
    This application is designed to forecast daily sales for Rossmann stores using historical data and store characteristics. Navigate through the app to explore the data, understand the models, and see the predictions in action.
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
        univariate_eda(train_df_merged) 

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

# EDA 2 Page
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

# Feature Engineering Page
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


# Modeling Page
elif pages == "Modeling":
    st.title("Modeling 🧠")

    if 'X_train' not in locals() or 'X_test' not in locals():
        st.markdown("Feature engineering and data splitting are being applied...")
        test_df_merged, train_df_merged, X_train, X_test, y_train, y_test = feature_engineering(train_df_merged, test_df_merged)

    tab1, tab2, tab3, tab4 = st.tabs(["Model Training", "Hyperparameter Tuning", "Model Performance", "Train on Full Dataset"])

    with tab1:
        st.subheader("Model Training")
        st.markdown("### Train the Final XGBoost Model")

        # Train the model using your function
        final_xgb_model = train_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

        # Check if the model was successfully trained and returned
        if final_xgb_model is not None:
            model_path = 'trained_xgb_model.pkl'  # Update this path as necessary
            joblib.dump(final_xgb_model['model'], model_path)
            st.write(f"Model saved to {model_path}")

            st.write(f"Model Performance Metrics: RMSE: {final_xgb_model['metrics']['rmse']}, MAE: {final_xgb_model['metrics']['mae']}, R2: {final_xgb_model['metrics']['r2']}")
        else:
            st.error("Model training failed. Please check the logs for more details.")

    with tab2:
        st.subheader("Hyperparameter Tuning")
        st.markdown("### Hyperparameter Tuning Results")

        # Perform hyperparameter tuning
        best_xgb_model, test_rmse_best_xgb, test_mae_best_xgb, test_r2_best_xgb, grid_search = tune_and_evaluate_xgboost(X_train, y_train, X_test, y_test)

        st.write("Best Parameters Found:", grid_search.best_params_)
        st.write("Best RMSE:", test_rmse_best_xgb)
        st.write("Best MAE:", test_mae_best_xgb)
        st.write("Best R-squared:", test_r2_best_xgb)

        # Save the best model from hyperparameter tuning
        tuned_model_path = 'best_xgb_model.pkl'
        joblib.dump(best_xgb_model, tuned_model_path)
        st.write(f"Tuned model saved to {tuned_model_path}")

    with tab3:
        st.subheader("Model Performance")
        st.markdown("### Model Performance Metrics")

        if final_xgb_model is not None:
            st.write("Final XGBoost Model Performance Metrics:")
            st.write(f"RMSE: {final_xgb_model['metrics']['rmse']}")
            st.write(f"MAE: {final_xgb_model['metrics']['mae']}")
            st.write(f"R2: {final_xgb_model['metrics']['r2']}")
        else:
            st.warning("Model performance metrics are unavailable. Please train the model first.")

    with tab4:
        st.subheader("Train on Full Dataset")
        st.markdown("### Save and Train the Model on the Full Dataset")

        # Train the model on the full dataset and save it
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

        model_path = '/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/best_xgb_model_full_trained.pkl'
        try:
            loaded_model = joblib.load(model_path)
            st.write(f"Model loaded from {model_path}")

            Test_df_future_sale = predict_sales_for_test_df(test_df_merged, model_path)
            # Display some predictions
            st.write("Here are some predictions:")
            st.write(Test_df_future_sale[['Date', 'Predicted_Sales']].head())

        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}. Please train and save a model first.")

    with tab2:
        st.subheader("Visualize Predictions")
        st.markdown("### Predicted vs Actual Sales")

        # Ensure 'Date' column is present in the dataframes
        if 'Date' in train_df_merged.columns and 'Date' in Test_df_future_sale.columns:
            plot_sales_comparison_streamlit(train_df_merged, Test_df_future_sale)
        else:
            st.warning("Date column missing in dataframes. Ensure the feature engineering step is completed correctly.")

# Conclusion Page
elif pages == "Conclusion":
    st.image('/path/to/conclusion/image.png', width=1000)  # Update this path

    st.title("Conclusion 🏁")
    st.markdown("""
    ### Key Takeaways
    - **Accurate Sales Predictions:** The model provides reliable predictions to assist Rossmann in sales planning.
    - **Actionable Insights:** The analysis identified key factors influencing sales, providing valuable insights for business decisions.
    
    ### Next Steps
    Consider using these predictions to optimize inventory, staffing, and promotional strategies across stores.
    """)

