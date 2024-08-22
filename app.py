import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from PIL import Image
import io
from dotenv import load_dotenv
import os

from datetime import datetime



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
# Set up the Streamlit app
st.set_page_config(page_title="Rossmann Sales Forecasting", layout="wide")

    # Load datasets function
    # @st.cache
def load_datasets():
    train_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/train.csv')
    test_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/test.csv')
    store_df = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/store.csv')
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'store_df': store_df
    }

data = load_datasets()
train_df = data['train_df']
store_df = data['store_df']
test_df = data['test_df']

    # Clean and merge datasets
train_df_merged, test_df_merged = clean_and_merge_datasets(train_df, test_df, store_df)

# Sidebar Navigation
st.sidebar.title("Rossmann Sales Forecasting App")
pages = st.sidebar.radio(
    "Navigate to",
    ["Introduction 🌟", "Exploratory Data Analysis (EDA) 📊", "Feature Engineering 🔧", "Modeling 🧠", "Forecast 🔮", "Recommendations 🚀", "Conclusion 🏁"]
)

# Authors section in sidebar
with st.sidebar.expander("About the Author"):
    st.sidebar.write("### Author")
    st.sidebar.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/Personal pic/Mustafa HS2.jpg', width=100)
    st.sidebar.write("Mustafa Aldabbas")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/your-linkedin-id/)")
    st.sidebar.write("### Bio")
    st.sidebar.write("""
        As a full-stack Data Analyst, I specialize in comprehensive data solutions from coding 
        in Python and SQL to advanced predictive modeling and interactive visualizations with Streamlit
        and Tableau. My experience spans detailed exploratory data analysis to deploying predictive models
        for retail forecasts, notably enhancing forecasting capabilities for major brands like Rossmann.
        I blend technical expertise with strategic insights to drive significant business outcomes in competitive markets.
    """)




if pages == "Introduction 🌟":
    st.markdown("<h1 style='color:white; text-align:center;'>Rossmann Sales Forecasting 📈</h1>", unsafe_allow_html=True)
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /rossmann.png', width=1000)  # Adjusted for better fit

    # Load the datasets
    train_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/train.csv')
    test_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/test.csv')
    store_data = pd.read_csv('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Datasets/Raw/store.csv')

    col1, col2 = st.columns([2, 1])  # Adjust column widths for better visual balance

    with col1:
        st.markdown("## Welcome to the Rossmann Sales Forecasting Project")
        st.markdown("""
        The project aims to address a critical business challenge: predicting daily sales for Rossmann stores. By leveraging historical sales data and various store-related features, we seek to build a robust predictive model to improve operational efficiency and optimize sales.
        """)

        st.markdown("### Project Definition and Objectives")
        st.markdown("""
        **Primary Objective:** Predict daily sales using historical data and store-specific features.
        **Secondary Objectives:** Understand sales influences, develop accurate forecasting models, and identify actionable insights to enhance performance.
        """)
        st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Project goals .pptx.png', width=1000)

    with col2:
        st.markdown("### Datasets Summary")
        st.markdown("**Store Dataset (`store.csv`)**")
        st.markdown("""
        - **Total Records:** 1,115
        - **Key Info:** Store type, assortment, competition distance, promotional details.
        - **Missing Values:** Some fields like competition distance and promotional timing details.
        """)

    st.markdown("### Data Previews")
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Project Objectibes .pptx.png', width=1000)

    tab1, tab2, tab3 = st.tabs(["Train Dataset", "Test Dataset", "Store Dataset"])

    
    with tab1:
        st.dataframe(train_data.head())
        st.markdown("""
        - **Total Records:** 1,017,209
        - **Features:** Store ID, day of week, date, sales, customers, open status, promotions.
        - **Missing Values:** None significant.
        """)
    
    with tab2:
        st.dataframe(test_data.head())
        st.markdown("""
        - **Total Records:** 41,088
        - **Features:** Similar to `train.csv` but without the sales column.
        - **Missing Values:** None significant.
        """)

    with tab3:
        st.dataframe(store_data.head())
        st.markdown("""
        - **Total Records:** 1,115
        - **Features:** Store type, assortment, competition distance, promo2 details.
        - **Missing Values:** Various, particularly in competition and promo2 details.
        """)

    st.markdown("""
    **Navigate through the app to explore the data, understand the models, and see the predictions in action!**
    """)

# EDA Page
elif pages == "Exploratory Data Analysis (EDA) 📊":
    st.title("Exploratory Data Analysis (EDA) 📊")
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Qoute.pptx.png', width=1000)


    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Univariate Analysis",
        "Bivariate Analysis",
        "Time Series Analysis",
        "Clustering Analysis",
        "Sales by Store Type and Assortment",
    ])

    with tab1:
        st.subheader("Univariate Analysis")
        
        # Create two columns for the first pair of images
        col1, col2 = st.columns(2)
        with col1:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/UNI variante /Sales distribution .png', width=330)
            st.markdown("""
            **Sales Distribution**
            - Displays the frequency of daily sales across all stores.
            - Most stores have sales clustering around lower values with some outliers.
            """)

        with col2:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/UNI variante /Customers Distribution.png', width=330)
            st.markdown("""
            **Customer Distribution**
            - Illustrates the count of customers visiting the stores.
            - Distribution shows a peak suggesting most stores have a similar range of daily customer counts.
            """)

        # Create two columns for the second pair of images
        col3, col4 = st.columns(2)
        with col3:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/UNI variante /Competition distiance distributino .png', width=330)
            st.markdown("""
            **Competition Distance Distribution**
            - Visualizes the distances to nearest competitors for each store.
            - A high number of stores have competitors within a short distance, indicating high competitive areas.
            """)

        with col4:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/UNI variante /Store type distribution .png', width=330)
            st.markdown("""
            **Store Type Distribution**
            - Counts of different store types within the chain.
            - Demonstrates dominance of one type over others, impacting sales strategies.
            """)


    with tab2:
        st.subheader("Bivariate Analysis")

        # Create columns to place images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Pomotions and sales .png', width=300)
            st.markdown("""
            **Average Sales with and without Promotions**
            - Clearly shows that promotions significantly boost sales.
            - Sales nearly double when promotions are active.
            """)
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Sales by competetion.png', width=300)
            st.markdown("""
            **Average Sales by Competition Distance**
            - Sales trends fluctuate with varying competition distances.
            - Notable spikes in sales at specific distance ranges, indicating strategic location advantages.
            """)

        with col2:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/sales by store .png', width=300)
            st.markdown("""
            **Average Sales by Store Type**
            - Shows how different store types perform in terms of sales.
            - Type 'a' stores generally perform better, indicating possibly larger or more centrally located stores.
            """)
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Customers vs sales.png', width=300)
            st.markdown("""
            **Customers vs Sales**
            - A strong positive correlation between the number of customers and sales.
            - Indicates that customer footfall is a significant driver of sales.
            """)

        # Single image with its own space
        st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Scool holidays .png', width=500)
        st.markdown("""
        **Average Sales on School Holidays vs Non-School Holidays**
        - Comparatively higher sales on school holidays.
        - Suggests an increase in shopping activity when schools are closed.
        """)


    with tab3:
        st.subheader("Time Series Analysis")

        # Creating columns to place images side by side
        col3, col4, col5 = st.columns(3)

        with col3:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Sales Trend over time .png', width=300)
            st.markdown("""
            **Weekly Sales Trend Over Time**
            - Displays the total weekly sales for the dataset over time.
            - Noticeable seasonal patterns and a general decline over the period.
            """)

        with col4:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Average sales by day .png', width=300)
            st.markdown("""
            **Average Sales by Day of the Week**
            - Shows average sales for each day of the week.
            - Highest sales typically occur in the middle of the week, with a sharp drop on Sundays.
            """)

        with col5:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/sales by month .png', width=300)
            st.markdown("""
            **Sales by Month**
            - Monthly sales distribution indicating variability and trends throughout the year.
            - Peaks often occur around mid-year and the holiday season, showing seasonal impact on sales.
            """)

    with tab4:
        st.subheader("Clustering Analysis")
        st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/sales cluster.png', width=500)
        st.markdown("""
        **Cluster Analysis: Sales, Customers, Promo, and Competition Distance**
        - This chart categorizes stores into clusters based on sales, customer traffic, promotional activity, and competition distance.
        - Highlights differences in sales and customer counts across clusters, showing how these factors correlate with promotional activities and proximity to competitors.
        """)

    with tab5:
        st.subheader("Sales by Store Type and Assortment")
        col1, col2 = st.columns(2)
        with col1:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/sales by store .png', width=300)
            st.markdown("""
            **Average Sales by Store Type**
            - Illustrates how different store types ('a', 'b', 'c', 'd') perform in terms of sales.
            - Type 'b' stores show notably higher sales, suggesting a more successful store format or location factor.
            """)
        with col2:
            st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Assortment .png', width=300)
            st.markdown("""
            **Average Sales by Assortment Type**
            - Compares sales performance across different assortment types ('a', 'b', 'c').
            - Shows that assortment type 'b' generally yields higher sales, indicating a possibly more appealing product mix to customers.
            """)





elif pages == "Feature Engineering 🔧":
    st.title("Feature Engineering 🔧")
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Feature engineering.pptx.png', width=1000)


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

elif pages == "Modeling 🧠":
    st.title("Modeling 🧠")
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Modeling.pptx.png', width=1000)



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
        st.markdown("""Random Forest Model Performance Metrics:

RMSE: 849.9031575766896

MAE: 581.953692501916

R2: 0.9376188668771028
""")


    with tab2:
        st.subheader("Train XGBoost")
        st.markdown("### Training XGBoost Model")
        st.markdown("""XGBoost Model Performance Metrics:

RMSE: 361.9634983167413

MAE: 247.00229878420924

R2: 0.9886852769647791"
""")

    with tab3:
        st.subheader("Tune XGBoost")
        st.markdown("### Hyperparameter Tuning for XGBoost")

        st.markdown(""" Hyperparameter Tuning for XGBoost
Best Parameters Found:

{
"colsample_bytree":0.8
"learning_rate":0.2
"max_depth":10
"n_estimators":200
"subsample":1
}
Best RMSE: 347.1120551397295

Best MAE: 236.93014566424435

Best R-squared: 0.989594720089926

Tuned model saved to best_xgb_model_tuned.pkl
                    """)

    with tab4:
        st.subheader("Model Performance")
        st.markdown("### Evaluate Model Performance")

        st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/EDA/Training models .png', width=1000)



    with tab5:
        st.subheader("Train Full Model")
        st.markdown("### Train Model on Full Dataset")

        st.button("Train Full Model")
            

# Prediction and Visualization Page
elif pages == "Forecast 🔮":
    st.title("Forecast 🔮")

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


elif pages == "Recommendations 🚀":
    st.title("Strategic Recommendations for Rossmann")
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /Reccomendations.png', width=1000)  # Update this path

    st.markdown("""
    ## Enhancing Sales Performance
    Based on our comprehensive analysis of sales data across various dimensions, we propose the following strategic initiatives to boost sales performance at Rossmann stores:
    """)

    st.header("1. Optimize Promotional Strategies")
    st.markdown("""
    - **Targeted Promotions:** Leverage insights from sales boosts during promotional periods to plan targeted promotions. Focus on store types and locations where promotions have historically led to significant sales uplifts.
    - **Promotion Timing:** Optimize the timing of promotions by aligning them with high-traffic days identified from the data, particularly mid-week, to maximize impact.
    """)
    
    st.header("2. Refine Assortment Planning")
    st.markdown("""
    - **Assortment Customization:** Tailor store assortments based on the sales performance of different categories. Prioritize high-performing assortments, particularly in store types where these assortments have shown to significantly drive sales.
    - **Local Preferences:** Analyze customer buying patterns to adapt the product offerings to local preferences and seasonal trends.
    """)

    st.header("3. Competitive Positioning")
    st.markdown("""
    - **Proximity to Competitors:** Implement strategic pricing and marketing campaigns in stores closer to competitors, as shown by the variable impact of competition distance on sales.
    - **Competitive Differentiation:** Enhance store features and customer service in highly competitive zones to differentiate Rossmann stores from nearby competitors.
    """)

    st.header("4. Enhance Customer Experience")
    st.markdown("""
    - **Store Layout Optimization:** Improve the layout and in-store navigation to enhance shopping experience, potentially increasing sales per customer visit.
    - **Customer Loyalty Programs:** Develop or enhance loyalty programs to increase repeat customer rates, drawing from insights on how customer counts correlate strongly with sales.
    """)

    st.header("5. Leverage Seasonal Trends")
    st.markdown("""
    - **Seasonal Marketing:** Capitalize on seasonal peaks by increasing inventory and marketing efforts during high-sales months and around holidays.
    - **Event-Based Promotions:** Use insights from sales performance during school holidays to plan events and promotions that attract more visitors during off-peak times.
    """)

    st.markdown("""
    ## Implementation
    These strategies should be implemented through a phased approach, starting with pilot programs in select store types and locations. Data-driven adjustments and scalability should be considered based on the initial impact assessments and feedback.
    """)

    st.markdown("""
    ## Conclusion
    By aligning our sales strategies with the detailed insights gained from our data analysis, Rossmann can effectively enhance overall sales performance, achieving a better market position and improved profitability.
    """)

elif pages == "Conclusion 🏁":
    # Header image for a polished look
    st.image('/Users/mustafaaldabbas/Documents/GitHub/Rossmann-sales-prediction/Visuals/streamlit pics /conclusion.png', width=1000)

    # Main Title with Emphasis
    st.title("Project Summary & Strategic Insights 🚀")

    # Introduction to the conclusion
    st.markdown("""
    ### Project Recap
    This project embarked on a journey to enhance Rossmann's sales strategies through data-driven insights. Here’s what we accomplished:
    """)

    # Using columns to separate sections visually
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("1. Key Sales Factors")
        st.markdown("""
        - **Promotions:** Significant boosts during promotional periods.
        - **Store Dynamics:** Diverse sales patterns across different store types.
        - **Market Competition:** Influence of proximity to competitors.
        """)
    with col2:
        st.subheader("2. Predictive Model Development")
        st.markdown("""
        - **Advanced Modeling:** Utilized Random Forest, XGBoost, and LightGBM.
        - **Accuracy Metrics:** Focused on RMSE and R² to measure performance.
        - **Best Model:** Highlighted the top-performing model for operational deployment.
        """)
    with col3:
        st.subheader("3. Actionable Insights")
        st.markdown("""
        - **Strategic Promotions:** Leveraged findings for targeted marketing initiatives.
        - **Customized Offerings:** Adapted strategies to fit unique store characteristics.
        - **Competitive Strategy:** Refined approaches based on competitor analysis.
        """)

    # Highlight Key Takeaways and Next Steps
    st.markdown("""
    ### Key Takeaways
    The analysis not only yielded precise sales forecasts but also unveiled critical sales drivers, providing Rossmann with the tools to refine its market approach.

    ### Future Directions
    Moving forward, the insights from this project will guide inventory management, staff planning, and promotional campaigns, ensuring that Rossmann remains competitive and proactive in its market strategies.
    """)

    # Visual Call to Action
    st.markdown("#### Explore Further")
    st.markdown("Navigate through the app to revisit data insights, review model performances, or adjust prediction parameters.")
