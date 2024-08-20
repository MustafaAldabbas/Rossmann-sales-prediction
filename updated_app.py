
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from functions import (
    load_datasets,
    clean_and_merge_datasets,
    plot_and_transform_sales,
    train_final_xgb_model,
    feature_engineering_and_split,
    hyperparameter_tuning_and_evaluate,
    correlation_heatmap,
    bivariate_eda,
    plot_sales_by_promo,
    plot_sales_by_competition_distance,
    perform_store_clustering,
    create_model_results_dataframe,
    create_features_and_predict_sales,
    plot_sales_comparison
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
st.sidebar.subheader("Navigate through different sections")

# Add navigation options
options = st.sidebar.radio(
    'Sections',
    ['Introduction', 'Data Exploration', 'Modeling', 'Predictions']
)

# Introduction Section
if options == 'Introduction':
    st.title("Rossmann Sales Forecasting")
    st.markdown(
        """
        This application is designed to forecast sales for the Rossmann pharmaceutical company. 
        Use the sidebar to navigate through the different sections, where you can explore data, 
        visualize trends, build predictive models, and generate sales forecasts.
        """
    )

# Data Exploration Section
elif options == 'Data Exploration':
    st.title("Exploratory Data Analysis")
    
    # Show some basic information about the datasets
    st.subheader("Training Data Overview")
    st.dataframe(train_df.head())
    
    st.subheader("Store Data Overview")
    st.dataframe(store_df.head())
    
    # Generate some visualizations
    st.subheader("Sales Distribution")
    plot_sales_by_promo(train_df_merged)
    st.pyplot()

    st.subheader("Sales by Competition Distance")
    plot_sales_by_competition_distance(train_df_merged)
    st.pyplot()

# Modeling Section
elif options == 'Modeling':
    st.title("Model Training and Evaluation")
    
    st.markdown("### Training the model...")
    # Example of training the model - you might want to include progress tracking
    model, evaluation_results = train_final_xgb_model(train_df_merged)
    
    st.subheader("Evaluation Results")
    st.write(evaluation_results)

# Predictions Section
elif options == 'Predictions':
    st.title("Generate Predictions")
    
    st.markdown("### Running predictions on the test set...")
    predictions = create_features_and_predict_sales(model, test_df_merged)
    
    st.subheader("Predictions Overview")
    st.dataframe(predictions.head())
    
    # Optionally, plot some of the predictions
    st.subheader("Sales Predictions")
    plot_sales_comparison(test_df_merged, predictions)
    st.pyplot()

st.sidebar.info("Developed by [Your Name]")
