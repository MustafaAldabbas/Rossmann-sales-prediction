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
pages = st.sidebar.radio(
    "Navigate to",
    ["Introduction", "EDA", "Feature Engineering", "Modeling", "Prediction", "Conclusion"]
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

    tab1, tab2, tab3, tab4 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Time Series Analysis", "Clustering Analysis"])

    with tab1:
        st.subheader("Univariate Analysis")
        st.markdown("### Sales Distribution")
        plot_and_transform_sales(train_df_merged, column='Sales')

    with tab2:
        st.subheader("Bivariate Analysis")
        st.markdown("### Correlation Heatmap")
        correlation_heatmap(train_df_merged)
        st.markdown("### Additional Bivariate Analysis")
        bivariate_eda(train_df_merged)

    with tab3:
        st.subheader("Time Series Analysis")
        st.markdown("### Sales Over Time")
        plot_sales_by_promo(train_df_merged)
        plot_sales_by_competition_distance(train_df_merged)

    with tab4:
        st.subheader("Clustering Analysis")
        st.markdown("### Store Clustering")
        perform_store_clustering(train_df_merged)

# Feature Engineering Page
elif pages == "Feature Engineering":
    st.title("Feature Engineering 🔧")

    st.markdown("### Applying Feature Engineering")
    train_df_merged, test_df_merged = feature_engineering_and_split(train_df_merged, test_df_merged)
    st.write(train_df_merged.head())
    st.write(test_df_merged.head())

    st.markdown("### Feature Engineering Completed")
    st.markdown("The dataset has been processed with new features.")

# Modeling Page
elif pages == "Modeling":
    st.title("Modeling 🧠")

    tab1, tab2, tab3 = st.tabs(["Model Training", "Hyperparameter Tuning", "Model Performance"])

    with tab1:
        st.subheader("Model Training")
        st.markdown("### Train the Final XGBoost Model")
        final_xgb_model = train_final_xgb_model(train_df_merged)
        st.markdown("#### Model Trained Successfully!")

        # Save the model
        model_path = '/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/trained_xgb_model.pkl'  # Update this path
        joblib.dump(final_xgb_model, model_path)
        st.write(f"Model saved to {model_path}")

    with tab2:
        st.subheader("Hyperparameter Tuning")
        st.markdown("### Hyperparameter Tuning Results")
        best_params, best_score = hyperparameter_tuning_and_evaluate(train_df_merged)
        st.write("Best Parameters:", best_params)
        st.write("Best Score:", best_score)

    with tab3:
        st.subheader("Model Performance")
        st.markdown("### Performance Metrics")
        model_results_df = create_model_results_dataframe(train_df_merged)
        st.write(model_results_df)

# Prediction Page
# Prediction Page
# Prediction Page
elif pages == "Prediction":
    st.title("Prediction 🔮")

    # Load the trained model
    final_xgb_model = joblib.load('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/trained_xgb_model.pkl')  # Update this path
    
    st.markdown("### Making Predictions")
    
    # Ensure that the feature engineering is applied correctly
    try:
        test_df_merged = create_features_and_predict_sales(train_df_merged, test_df_merged, final_xgb_model)
        st.write(test_df_merged[['Store', 'Date', 'Predicted_Sales']].head())
        st.markdown("#### Actual vs Predicted Sales")
        plot_sales_comparison(train_df_merged, test_df_merged)
    except KeyError as e:
        st.error(f"Error: {str(e)}")



# Conclusion Page
elif pages == "Conclusion":
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /conclusion.png', width=1000)  # Update this path

    st.title("Conclusion 🏁")
    st.markdown("""
    ### Key Takeaways
    - **Accurate Sales Predictions:** The model provides reliable predictions to assist Rossmann in sales planning.
    - **Actionable Insights:** The analysis identified key factors influencing sales, providing valuable insights for business decisions.
    
    ### Next Steps
    Consider using these predictions to optimize inventory, staffing, and promotional strategies across stores.
    """)
