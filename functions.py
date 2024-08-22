# Data manipulation and processing
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import yaml
# Utility libraries
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import streamlit as st

import sys

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
#Loding the Datasets

import yaml
import pandas as pd

def load_datasets(config_path='config.yaml'):
    """
    Load datasets based on the paths specified in the config.yaml file.

    Args:
        config_path (str): Path to the configuration file. Defaults to 'config.yaml'.

    Returns:
        dict: A dictionary containing the train, store, and test DataFrames.
    """
    # Load the config.yaml file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Access paths from the config.yaml file
    train_data_path = config['paths']['train_data']
    store_data_path = config['paths']['store_data']
    test_data_path = config['paths']['test_data']

    # Load the datasets using paths from config.yaml
    train_df = pd.read_csv(train_data_path)
    store_df = pd.read_csv(store_data_path)
    test_df = pd.read_csv(test_data_path)

    # Return the DataFrames as a dictionary
    return {
        'train_df': train_df,
        'store_df': store_df,
        'test_df': test_df
    }

 #-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
 


def clean_and_merge_datasets(train_df, test_df, store_df):
    """
    Cleans and merges the train, test, and store datasets.
    
    Steps:
    1. Convert date columns to datetime format.
    2. Ensure categorical variables are properly formatted.
    3. Remove outliers using the IQR method.
    4. Merge train and test datasets with store data.
    5. Replace missing values with 0 for numeric columns.
    6. Visualize the cleaned and merged datasets.
    7. Save the cleaned and merged datasets to CSV files.

    Args:
        train_df (pd.DataFrame): Training dataset.
        test_df (pd.DataFrame): Testing dataset.
        store_df (pd.DataFrame): Store information dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: Cleaned and merged train and test datasets.
    """
    
    # Step 1: Convert Date columns to datetime format
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    # Step 2: Ensure categorical variables are properly formatted
    categorical_columns = {
        'train_df': ['StateHoliday', 'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday'],
        'store_df': ['Store', 'StoreType', 'Assortment', 'PromoInterval'],
        'test_df': ['StateHoliday', 'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday']
    }

    for column in categorical_columns['train_df']:
        train_df[column] = train_df[column].astype('category')

    for column in categorical_columns['store_df']:
        store_df[column] = store_df[column].astype('category')

    for column in categorical_columns['test_df']:
        test_df[column] = test_df[column].astype('category')

    # Step 3: Identify Outliers using IQR method
    Q1 = train_df['Sales'].quantile(0.25)
    Q3 = train_df['Sales'].quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove Outliers
    train_df_cleaned = train_df[(train_df['Sales'] >= lower_bound) & (train_df['Sales'] <= upper_bound)]

    # Step 4: Merge train_df with store_df
    train_df_merged = pd.merge(train_df_cleaned, store_df, on='Store', how='left')
    test_df_merged = pd.merge(test_df, store_df, on='Store', how='left')

    # Step 5: Replace missing values with 0 for numeric columns only
    numeric_columns_train = train_df_merged.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns_test = test_df_merged.select_dtypes(include=['float64', 'int64']).columns

    train_df_merged[numeric_columns_train] = train_df_merged[numeric_columns_train].fillna(0)
    test_df_merged[numeric_columns_test] = test_df_merged[numeric_columns_test].fillna(0)

    # Step 6: Visualize the Sales Data after cleaning and merging
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(train_df_merged['Sales'], bins=50, color='blue', edgecolor='black')
    plt.title('Sales Distribution After Cleaning')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')

    # Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(train_df_merged['Sales'], vert=False)
    plt.title('Boxplot of Sales After Cleaning')

    plt.tight_layout()
    plt.show()

    # Display the first few rows of the merged dataframes to confirm the merge
    print("\nMerged train_df:")
    print(train_df_merged.head())

    print("\nMerged test_df:")
    print(test_df_merged.head())

    # Step 7: Save the cleaned and merged datasets to CSV files
    train_df_merged.to_csv('train_df_merged_clean.csv', index=False)
    test_df_merged.to_csv('test_df_merged_clean.csv', index=False)

    return train_df_merged, test_df_merged



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_and_transform_sales(df, column='Sales'):
    """
    Plots the distribution of the sales data and applies a log transformation.

    Args:
        df (pd.DataFrame): The DataFrame containing the sales data.
        column (str): The name of the column to be transformed. Defaults to 'Sales'.

    Returns:
        pd.DataFrame: The DataFrame with the transformed sales column.
    """
    # Plot the distribution of the target variable before transformation
    plt.figure(figsize=(8, 3))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()  # Display the plot in Jupyter Notebook

    # Apply log transformation
    df[column] = np.log1p(df[column])

    # Plot the distribution after log transformation
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of Log-Transformed {column}')
    plt.tight_layout()
    plt.show()  # Display the plot in Jupyter Notebook

    return df



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import seaborn as sns

def univariate_edaN(df):
    """
    Performs univariate exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots the distributions of Sales, Customers, StoreType, and CompetitionDistance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Set up the figure and axis for subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 12))

    # Sales distribution
    sns.histplot(df['Sales'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Sales')

    # Customers distribution
    sns.histplot(df['Customers'], bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Customers')

    # StoreType distribution
    sns.countplot(x='StoreType', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Count of Store Types')

    # CompetitionDistance distribution
    sns.histplot(df['CompetitionDistance'], bins=50, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Competition Distance')

    # Adjust layout
    plt.tight_layout()

    # Display the figure in Jupyter Notebook
    plt.show()



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
def univariate_edast(df):
    """
    Performs univariate exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots the distributions of Sales, Customers, StoreType, and CompetitionDistance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
    
    Returns:
        matplotlib.figure.Figure: The figure object containing all the subplots.
    """
    # Set up the figure and axis for subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # Sales distribution
    sns.histplot(df['Sales'], bins=50, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Sales')

    # Customers distribution
    sns.histplot(df['Customers'], bins=50, kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Customers')

    # StoreType distribution
    sns.countplot(x='StoreType', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Count of Store Types')

    # CompetitionDistance distribution
    sns.histplot(df['CompetitionDistance'], bins=50, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Competition Distance')

    # Adjust layout
    plt.tight_layout()

    # Return the figure object
    return fig

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def bivariate_eda(df):
    """
    Performs bivariate exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots the relationships between Sales and other variables such as Promo, StoreType, and CompetitionDistance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Sales vs Promo
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales vs Promo')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales vs StoreType
    plt.figure(figsize=(5, 3))
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales vs StoreType')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales vs CompetitionDistance
    plt.figure(figsize=(5, 3))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Sales vs Competition Distance')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

def bivariate_edaN(df):
    """
    Performs bivariate exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots the relationships between Sales and other variables such as Promo, StoreType, and CompetitionDistance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Sales vs Promo
    plt.figure(figsize=(7, 3))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales vs Promo')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales vs StoreType
    plt.figure(figsize=(7, 3))
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales vs StoreType')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales vs CompetitionDistance
    plt.figure(figsize=(7, 3))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Sales vs Competition Distance')
    plt.tight_layout()
    
# Display the figure in Jupyter Notebook
    plt.show()
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
def timeseries_edaN(df, date_col='Date', sales_col='Sales'):
    """
    Performs time series exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots sales trends over time, by day of the week, and by month.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        date_col (str): The name of the date column in the DataFrame. Defaults to 'Date'.
        sales_col (str): The name of the sales column in the DataFrame. Defaults to 'Sales'.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Sales over time
    plt.figure(figsize=(8, 5))
    df.groupby(date_col)[sales_col].sum().plot()
    plt.title('Total Sales Over Time')
    plt.ylabel(sales_col)
    plt.tight_layout()
    plt.show()


    # Sales by DayOfWeek
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='DayOfWeek', y=sales_col, data=df)
    plt.title('Sales by Day of the Week')
    plt.tight_layout()
    plt.show()


    # Sales by Month
    df['Month'] = df[date_col].dt.month
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Month', y=sales_col, data=df)
    plt.title('Sales by Month')
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def timeseries_eda(df, date_col='Date', sales_col='Sales'):
    """
    Performs time series exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots sales trends over time, by day of the week, and by month.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        date_col (str): The name of the date column in the DataFrame. Defaults to 'Date'.
        sales_col (str): The name of the sales column in the DataFrame. Defaults to 'Sales'.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Sales over time
    plt.figure(figsize=(14, 7))
    df.groupby(date_col)[sales_col].sum().plot()
    plt.title('Total Sales Over Time')
    plt.ylabel(sales_col)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales by DayOfWeek
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DayOfWeek', y=sales_col, data=df)
    plt.title('Sales by Day of the Week')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Sales by Month
    df['Month'] = df[date_col].dt.month
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Month', y=sales_col, data=df)
    plt.title('Sales by Month')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_promoN(df):
    """
    This function takes a DataFrame and plots the average sales for promo vs non-promo days.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'Promo' and 'Sales' columns.

    Returns:
    None: Displays a bar chart of average sales with and without promotions.
    """
    # Calculate average sales for promo vs non-promo days
    sales_by_promo = df.groupby('Promo')['Sales'].mean()

    # Plotting the sales by promotion
    plt.figure(figsize=(4, 3))
    sns.barplot(x='Promo', y='Sales', data=df, ci=None, palette='Set2')
    plt.xlabel('Promotion (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Promotion')
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_promo(df):
    """
    This function takes a DataFrame and plots the average sales for promo vs non-promo days.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'Promo' and 'Sales' columns.

    Returns:
    None: Displays a bar chart of average sales with and without promotions.
    """
    # Calculate average sales for promo vs non-promo days
    sales_by_promo = df.groupby('Promo')['Sales'].mean()

    # Plotting the sales by promotion
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Promo', y='Sales', data=df, ci=None, palette='Set2')
    plt.xlabel('Promotion (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Promotion')
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit





#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_competition_distanceN(df, bins=10):
    """
    This function takes a DataFrame and plots the average sales against competition distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'CompetitionDistance' and 'Sales' columns.
    bins (int): Number of bins to divide the competition distance into. Default is 10.

    Returns:
    None: Displays a line chart of average sales by competition distance.
    """
    # Check if the required columns exist in the DataFrame
    if 'CompetitionDistance' not in df.columns or 'Sales' not in df.columns:
        st.error("The required columns ('CompetitionDistance', 'Sales') are not present in the DataFrame.")
        return
    
    # Drop rows with missing values in the relevant columns
    df = df.dropna(subset=['CompetitionDistance', 'Sales'])

    # Analyze sales against competition distance
    try:
        sales_by_competition_distance = df.groupby(pd.cut(df['CompetitionDistance'], bins=bins))['Sales'].mean()
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return

    # Plotting the sales by competition distance
    plt.figure(figsize=(6, 4))
    sales_by_competition_distance.plot(kind='line', marker='o', color='green')
    plt.xlabel('Competition Distance (binned)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Competition Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_competition_distance(df, bins=10):
    """
    This function takes a DataFrame and plots the average sales against competition distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'CompetitionDistance' and 'Sales' columns.
    bins (int): Number of bins to divide the competition distance into. Default is 10.

    Returns:
    None: Displays a line chart of average sales by competition distance.
    """
    # Check if the required columns exist in the DataFrame
    if 'CompetitionDistance' not in df.columns or 'Sales' not in df.columns:
        st.error("The required columns ('CompetitionDistance', 'Sales') are not present in the DataFrame.")
        return
    
    # Drop rows with missing values in the relevant columns
    df = df.dropna(subset=['CompetitionDistance', 'Sales'])

    # Analyze sales against competition distance
    try:
        sales_by_competition_distance = df.groupby(pd.cut(df['CompetitionDistance'], bins=bins))['Sales'].mean()
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return

    # Plotting the sales by competition distance
    plt.figure(figsize=(8, 6))
    sales_by_competition_distance.plot(kind='line', marker='o', color='green')
    plt.xlabel('Competition Distance (binned)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Competition Distance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit




#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

def plot_sales_by_school_holidayN(df):
    """
    This function takes a DataFrame and plots the average sales on school holidays versus non-school holidays.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'SchoolHoliday' and 'Sales' columns.

    Returns:
    None: Displays a bar chart of average sales on school holidays vs non-school holidays.
    """
    # Calculate average sales on school holidays vs non-school holidays
    sales_by_school_holiday = df.groupby('SchoolHoliday')['Sales'].mean()

    # Plotting the average sales on school holidays vs non-school holidays
    plt.figure(figsize=(6, 4))
    sales_by_school_holiday.plot(kind='bar', color='purple')
    plt.xlabel('School Holiday (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales on School Holidays vs Non-School Holidays')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_school_holiday(df):
    """
    This function takes a DataFrame and plots the average sales on school holidays versus non-school holidays.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'SchoolHoliday' and 'Sales' columns.

    Returns:
    None: Displays a bar chart of average sales on school holidays vs non-school holidays.
    """
    # Calculate average sales on school holidays vs non-school holidays
    sales_by_school_holiday = df.groupby('SchoolHoliday')['Sales'].mean()

    # Plotting the average sales on school holidays vs non-school holidays
    plt.figure(figsize=(8, 6))
    sales_by_school_holiday.plot(kind='bar', color='purple')
    plt.xlabel('School Holiday (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales on School Holidays vs Non-School Holidays')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

def plot_sales_by_store_type_and_assortmentN(df):
    """
    This function takes a DataFrame and plots the average sales by store type and assortment type.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'StoreType', 'Assortment', and 'Sales' columns.

    Returns:
    None: Displays bar charts of average sales by store type and by assortment type.
    """
    # Calculate average sales by store type
    sales_by_store_type = df.groupby('StoreType')['Sales'].mean()

    # Plotting the average sales by store type
    plt.figure(figsize=(6,4))
    sales_by_store_type.plot(kind='bar', color='teal')
    plt.xlabel('Store Type')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Store Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Calculate average sales by assortment
    sales_by_assortment = df.groupby('Assortment')['Sales'].mean()

    # Plotting the average sales by assortment
    plt.figure(figsize=(6, 4))
    sales_by_assortment.plot(kind='bar', color='coral')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Assortment Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import matplotlib.pyplot as plt
import streamlit as st

def plot_sales_by_store_type_and_assortment(df):
    """
    This function takes a DataFrame and plots the average sales by store type and assortment type.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'StoreType', 'Assortment', and 'Sales' columns.

    Returns:
    None: Displays bar charts of average sales by store type and by assortment type.
    """
    # Calculate average sales by store type
    sales_by_store_type = df.groupby('StoreType')['Sales'].mean()

    # Plotting the average sales by store type
    plt.figure(figsize=(8, 6))
    sales_by_store_type.plot(kind='bar', color='teal')
    plt.xlabel('Store Type')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Store Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit

    # Calculate average sales by assortment
    sales_by_assortment = df.groupby('Assortment')['Sales'].mean()

    # Plotting the average sales by assortment
    plt.figure(figsize=(8, 6))
    sales_by_assortment.plot(kind='bar', color='coral')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Assortment Type')
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)  # Display the plot in Streamlit


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_store_clusteringN(df, n_clusters=3, max_clusters=10):
    """
    This function performs K-means clustering on store data based on sales, customers, promo participation, and competition distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing store data, with 'Store', 'Sales', 'Customers', 'Promo', and 'CompetitionDistance' columns.
    n_clusters (int): Number of clusters to use for K-means clustering. Default is 3.
    max_clusters (int): Maximum number of clusters to test in the Elbow Method. Default is 10.

    Returns:
    pd.DataFrame: DataFrame with cluster labels assigned to each store.
    None: Displays the Elbow Method plot and cluster analysis bar charts.
    """
    # Ensure relevant columns are numeric
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Customers'] = pd.to_numeric(df['Customers'], errors='coerce')
    df['Promo'] = pd.to_numeric(df['Promo'], errors='coerce')
    df['CompetitionDistance'] = pd.to_numeric(df['CompetitionDistance'], errors='coerce')

    # Select relevant features for clustering
    store_clustering_data = df.groupby('Store').agg({
        'Sales': 'mean',
        'Customers': 'mean',
        'Promo': 'mean',
        'CompetitionDistance': 'mean'
    }).reset_index()

    # Handle any missing values by filling with 0
    store_clustering_data.fillna(0, inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(store_clustering_data[['Sales', 'Customers', 'Promo', 'CompetitionDistance']])

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method chart
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()  # Display the plot in Jupyter Notebook

    # Applying K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add the cluster labels to the original store data
    store_clustering_data['Cluster'] = clusters

    # Analyze the characteristics of each cluster by calculating the mean of each feature within clusters
    cluster_analysis = store_clustering_data.groupby('Cluster').mean()

    # Display the cluster analysis in Jupyter Notebook
    print("Cluster Analysis:")
    print(cluster_analysis)

    # Plotting the cluster centers for visualization
    plt.figure(figsize=(6, 3))
    cluster_analysis[['Sales', 'Customers', 'Promo', 'CompetitionDistance']].plot(kind='bar')
    plt.title('Cluster Analysis: Sales, Customers, Promo, and Competition Distance by Cluster')
    plt.xticks(rotation=0)
    plt.show()  # Display the plot in Jupyter Notebook

    return store_clustering_data

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def perform_store_clustering(df, n_clusters=3, max_clusters=10):
    """
    This function performs K-means clustering on store data based on sales, customers, promo participation, and competition distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing store data, with 'Store', 'Sales', 'Customers', 'Promo', and 'CompetitionDistance' columns.
    n_clusters (int): Number of clusters to use for K-means clustering. Default is 3.
    max_clusters (int): Maximum number of clusters to test in the Elbow Method. Default is 10.

    Returns:
    pd.DataFrame: DataFrame with cluster labels assigned to each store.
    None: Displays the Elbow Method plot and cluster analysis bar charts.
    """
    # Ensure relevant columns are numeric
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df['Customers'] = pd.to_numeric(df['Customers'], errors='coerce')
    df['Promo'] = pd.to_numeric(df['Promo'], errors='coerce')
    df['CompetitionDistance'] = pd.to_numeric(df['CompetitionDistance'], errors='coerce')

    # Select relevant features for clustering
    store_clustering_data = df.groupby('Store').agg({
        'Sales': 'mean',
        'Customers': 'mean',
        'Promo': 'mean',
        'CompetitionDistance': 'mean'
    }).reset_index()

    # Handle any missing values by filling with 0
    store_clustering_data.fillna(0, inplace=True)

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(store_clustering_data[['Sales', 'Customers', 'Promo', 'CompetitionDistance']])

    # Determine the optimal number of clusters using the Elbow Method
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow Method chart
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Number of Clusters')
    st.pyplot(plt)  # Display the plot in Streamlit

    # Applying K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add the cluster labels to the original store data
    store_clustering_data['Cluster'] = clusters

    # Analyze the characteristics of each cluster by calculating the mean of each feature within clusters
    cluster_analysis = store_clustering_data.groupby('Cluster').mean()

    # Display the cluster analysis using Streamlit
    st.write("Cluster Analysis:")
    st.write(cluster_analysis)

    # Plotting the cluster centers for visualization
    plt.figure(figsize=(12, 6))
    cluster_analysis[['Sales', 'Customers', 'Promo', 'CompetitionDistance']].plot(kind='bar')
    plt.title('Cluster Analysis: Sales, Customers, Promo, and Competition Distance by Cluster')
    plt.xticks(rotation=0)
    st.pyplot(plt)  # Display the plot in Streamlit

    return store_clustering_data





#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering(train_df_merged, test_df_merged):
    # Extract date-related features
    for df in [train_df_merged, test_df_merged]:
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    print("Date-related features added:", train_df_merged.columns)

    # Create binary features for holidays and promotions
    for df in [train_df_merged, test_df_merged]:
        df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
        df['IsPromo'] = df['Promo'].apply(lambda x: 1 if x == 1 else 0)

    print("Binary features added:", train_df_merged.columns)

    # Calculate competition duration in months
    for df in [train_df_merged, test_df_merged]:
        df['CompetitionOpenSince'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: max(x, 0))

    print("Competition features added:", train_df_merged.columns)

    # Calculate Promo2 duration in weeks
    for df in [train_df_merged, test_df_merged]:
        df['Promo2OpenSince'] = 52 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek'])
        df['Promo2OpenSince'] = df['Promo2OpenSince'].apply(lambda x: max(x, 0))

    print("Promo2 features added:", train_df_merged.columns)

    # Label Encoding for 'StateHoliday'
    label_encoder = LabelEncoder()
    for df in [train_df_merged, test_df_merged]:
        df['StateHoliday'] = df['StateHoliday'].astype(str)
        df['StateHoliday'] = label_encoder.fit_transform(df['StateHoliday'])

    print("Label encoding applied:", train_df_merged.columns)

    # One-Hot Encoding for categorical variables (check if the columns exist)
    for col in ['StoreType', 'Assortment', 'PromoInterval']:
        if col in train_df_merged.columns:
            train_df_merged = pd.get_dummies(train_df_merged, columns=[col], drop_first=True)
        if col in test_df_merged.columns:
            test_df_merged = pd.get_dummies(test_df_merged, columns=[col], drop_first=True)

    # Align columns between train and test sets
    test_df_merged = test_df_merged.reindex(columns=train_df_merged.columns, fill_value=0)
    print("One-hot encoding applied and columns aligned:", train_df_merged.columns)

    # Lag Features for Sales
    for lag in [1, 7, 30]:
        train_df_merged[f'Sales_Lag_{lag}'] = train_df_merged['Sales'].shift(lag)
        test_df_merged[f'Sales_Lag_{lag}'] = train_df_merged['Sales'].shift(lag)
    
    print("Lag features for Sales added:", train_df_merged.columns)

    # Create moving average features for 'Sales' and 'Customers'
    for window in [7, 30]:
        train_df_merged[f'Sales_MA_{window}'] = train_df_merged['Sales'].rolling(window=window).mean()
        train_df_merged[f'Customers_MA_{window}'] = train_df_merged['Customers'].rolling(window=window).mean()

        test_df_merged[f'Sales_MA_{window}'] = train_df_merged['Sales'].rolling(window=window).mean()
        test_df_merged[f'Customers_MA_{window}'] = train_df_merged['Customers'].rolling(window=window).mean()

    print("Moving average features added:", train_df_merged.columns)

    # Sine and Cosine Transformations for Cyclical Features
    for df in [train_df_merged, test_df_merged]:
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)

    print("Sine and cosine transformations added:", train_df_merged.columns)

    # Handle NaN values created by lag and rolling operations
    train_df_merged.dropna(inplace=True)
    test_df_merged.fillna(0, inplace=True)

    print("Final columns after NaN handling:", train_df_merged.columns)
    
    # Prepare features and target
    feature_cols = [col for col in train_df_merged.columns if col not in ['Sales', 'Date']]
    target_col = 'Sales'
    
    X = train_df_merged[feature_cols]
    y = train_df_merged[target_col]
    
    # Ensure the DataFrame is sorted by date
    train_df_merged = train_df_merged.sort_values(by='Date')
    
    # Split the data chronologically
    split_date = train_df_merged['Date'].max() - pd.DateOffset(months=6)  # 6 months before the last date
    train_df = train_df_merged[train_df_merged['Date'] < split_date]
    test_df = train_df_merged[train_df_merged['Date'] >= split_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return test_df_merged, train_df_merged, X_train, X_test, y_train, y_test




#-----------------------------------------------------------------------------------#


def train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10, random_state=42):
    """
    Trains a Random Forest model and evaluates it on the test set.
    
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.
        n_estimators (int): Number of trees in the forest. Defaults to 100.
        max_depth (int): Maximum depth of the tree. Defaults to 10.
        random_state (int): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        dict: A dictionary containing the model, predictions, and evaluation metrics (RMSE, MAE, R-squared).
    """
    # Initialize the RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    
    # Train the model on the full training dataset
    rf_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_test_pred = rf_model.predict(X_test)
    
    # Evaluate model performance on the test set
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Print the evaluation metrics
    print(f"Test RMSE: {test_rmse}")
    print(f"Test MAE: {test_mae}")
    print(f"Test R-squared: {test_r2}")
    
    # Return the model and evaluation metrics
    return {
        'model': rf_model,
        'predictions': y_test_pred,
        'metrics': {
            'rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2
        }
    }



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


def plot_rf_feature_importance(model, X_train, top_n=20):
    """
    Plots the feature importance of a trained Random Forest model.
    
    Args:
        model (RandomForestRegressor): A trained Random Forest model.
        X_train (pd.DataFrame): The training features used to train the model.
        top_n (int): The number of top features to display. Defaults to 20.
    
    Returns:
        pd.DataFrame: A DataFrame containing the features and their importances.
    """
    # Get feature importances from the Random Forest model
    importances_rf = model.feature_importances_
    features_rf = X_train.columns

    # Create a DataFrame for visualization
    feature_importance_rf_df = pd.DataFrame({'Feature': features_rf, 'Importance': importances_rf}).sort_values(by='Importance', ascending=False)

    # Display the top n most important features for Random Forest
    print(f"Random Forest Feature Importance (Top {top_n}):")
    print(feature_importance_rf_df.head(top_n))

    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_rf_df['Feature'].head(top_n), feature_importance_rf_df['Importance'].head(top_n))
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Important Features - Random Forest")
    plt.gca().invert_yaxis()
    plt.show()

    return feature_importance_rf_df



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


def train_and_evaluate_xgboost(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10, random_state=42, enable_categorical=True):
    """
    Trains an XGBoost model and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.
        n_estimators (int): Number of boosting rounds. Defaults to 100.
        max_depth (int): Maximum depth of the trees. Defaults to 10.
        random_state (int): Random seed for reproducibility. Defaults to 42.
        enable_categorical (bool): Whether to enable categorical support. Defaults to True.

    Returns:
        dict: A dictionary containing the model, predictions, and evaluation metrics (RMSE, MAE, R-squared).
    """
    # Initialize the XGBoost Regressor
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, enable_categorical=enable_categorical)
    
    try:
        # Train the model on the training data
        xgb_model.fit(X_train, y_train)

        # Predict on the test set
        y_test_pred_xgb = xgb_model.predict(X_test)

        # Evaluate model performance on the test set
        test_rmse_xgb = mean_squared_error(y_test, y_test_pred_xgb, squared=False)
        test_mae_xgb = mean_absolute_error(y_test, y_test_pred_xgb)
        test_r2_xgb = r2_score(y_test, y_test_pred_xgb)

        # Print the evaluation metrics
        print(f"XGBoost Test RMSE: {test_rmse_xgb}")
        print(f"XGBoost Test MAE: {test_mae_xgb}")
        print(f"XGBoost Test R-squared: {test_r2_xgb}")

        # Return the model and evaluation metrics
        return {
            'model': xgb_model,
            'predictions': y_test_pred_xgb,
            'metrics': {
                'rmse': test_rmse_xgb,
                'mae': test_mae_xgb,
                'r2': test_r2_xgb
            }
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None




#-----------------------------------------------------------------------------------

def tune_and_evaluate_xgboost(X_train, y_train, X_test, y_test):
    # Define the model
    xgb_model = xgb.XGBRegressor(random_state=42, enable_categorical=True)

    # Define the grid of hyperparameters
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'learning_rate': [ 0.2],
        'subsample': [1.0],
        'colsample_bytree': [0.8]
    }

    # Set up the grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring='neg_root_mean_squared_error', 
                               cv=3, verbose=2, n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best RMSE: ", -grid_search.best_score_)

    # Train the best model
    best_xgb_model = grid_search.best_estimator_

    # Predict on the test set
    y_test_pred_best_xgb = best_xgb_model.predict(X_test)

    # Evaluate the best model on the test set
    test_rmse_best_xgb = mean_squared_error(y_test, y_test_pred_best_xgb, squared=False)
    test_mae_best_xgb = mean_absolute_error(y_test, y_test_pred_best_xgb)
    test_r2_best_xgb = r2_score(y_test, y_test_pred_best_xgb)

    print(f"Best XGBoost Test RMSE: {test_rmse_best_xgb}")
    print(f"Best XGBoost Test MAE: {test_mae_best_xgb}")
    print(f"Best XGBoost Test R-squared: {test_r2_best_xgb}")

    return best_xgb_model, test_rmse_best_xgb, test_mae_best_xgb, test_r2_best_xgb, grid_search




#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
def create_display_and_visualize_model_resultsN():
    """
    Creates a DataFrame summarizing the results of different models, displays it, and visualizes the results.
    
    Returns:
    - results_df: A DataFrame summarizing the performance and hyperparameters of each model.
    """
    
    # Define the results for each model using the results you shared
    model_results = {
        'Model': ['Random Forest', 'XGBoost', 'XGBoost (Tuned)'],
        'RMSE': [0.44946992012881254, 0.29817999295162884, 0.2892128493088795],
        'MAE': [0.22129262422877377, 0.07588247725820102, 0.07769083766709649],
        'R-squared': [0.9787274724229729, 0.9906378664909703, 0.9911924929785798]
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame(model_results)

    # Display the DataFrame in Streamlit
    st.write("### Model Performance Results")
    st.write(results_df)

    ### Plot: Comparison of RMSE, MAE, and R-squared for the models
    fig, ax1 = plt.subplots(figsize=(8, 6))

    bar_width = 0.2
    index = range(3)

    # Plot RMSE and MAE on the first y-axis
    bars1 = ax1.bar([i - bar_width/2 for i in index], results_df['RMSE'], bar_width, label='RMSE', color='skyblue')
    bars2 = ax1.bar([i + bar_width/2 for i in index], results_df['MAE'], bar_width, label='MAE', color='lightcoral')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE / MAE')
    ax1.set_title('Model Performance Comparison (Random Forest, XGBoost, XGBoost Tuned)')
    ax1.set_xticks(index)
    ax1.set_xticklabels(results_df['Model'])

    # Create a second y-axis for R-squared
    ax2 = ax1.twinx()
    bars3 = ax2.bar([i + 1.5 * bar_width for i in index], results_df['R-squared'], bar_width, label='R-squared', color='lightgreen')
    ax2.set_ylabel('R-squared')

    # Combine legends from both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

    return results_df


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def create_display_and_visualize_model_results():
    """
    Creates a DataFrame summarizing the results of different models, displays it, and visualizes the results.
    
    Returns:
    - results_df: A DataFrame summarizing the performance and hyperparameters of each model.
    """
    
    # Define the results for each model using the results you shared
    model_results = {
        'Model': ['Random Forest', 'XGBoost', 'XGBoost (Tuned)'],
        'RMSE': [0.44946992012881254, 0.29817999295162884, 0.2892128493088795],
        'MAE': [0.22129262422877377, 0.07588247725820102, 0.07769083766709649],
        'R-squared': [0.9787274724229729, 0.9906378664909703, 0.9911924929785798]
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame(model_results)

    # Display the DataFrame in Streamlit
    st.write("### Model Performance Results")
    st.write(results_df)

    ### Plot: Comparison of RMSE, MAE, and R-squared for the models
    fig, ax1 = plt.subplots(figsize=(8, 6))

    bar_width = 0.2
    index = range(3)

    # Plot RMSE and MAE on the first y-axis
    bars1 = ax1.bar([i - bar_width/2 for i in index], results_df['RMSE'], bar_width, label='RMSE', color='skyblue')
    bars2 = ax1.bar([i + bar_width/2 for i in index], results_df['MAE'], bar_width, label='MAE', color='lightcoral')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE / MAE')
    ax1.set_title('Model Performance Comparison (Random Forest, XGBoost, XGBoost Tuned)')
    ax1.set_xticks(index)
    ax1.set_xticklabels(results_df['Model'])

    # Create a second y-axis for R-squared
    ax2 = ax1.twinx()
    bars3 = ax2.bar([i + 1.5 * bar_width for i in index], results_df['R-squared'], bar_width, label='R-squared', color='lightgreen')
    ax2.set_ylabel('R-squared')

    # Combine legends from both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot in Streamlit
    st.pyplot(fig)

    return results_df

#-----------------------------------------------------------------------------------#
import joblib
import xgboost as xgb

def save_and_train_full_model(best_xgb_model, train_df_merged, tuned_model_path='best_xgb_model_tuned.pkl', full_model_path='best_xgb_model_full_trained.pkl'):
    """
    Saves the tuned XGBoost model and trains it on the full dataset.
    
    Parameters:
    - best_xgb_model: The best XGBoost model found after hyperparameter tuning.
    - train_df_merged: The full training dataset after feature engineering.
    - tuned_model_path: Path where the tuned model will be saved.
    - full_model_path: Path where the fully trained model will be saved.
    
    Returns:
    - best_xgb_model: The best XGBoost model after tuning.
    - best_xgb_model_full: The XGBoost model trained on the full dataset.
    """
    
    # Save the tuned XGBoost model
    joblib.dump(best_xgb_model, tuned_model_path)
    
    # Load the saved tuned model
    best_xgb_model_full = joblib.load(tuned_model_path)
    
    # Prepare the training data by dropping the specified columns
    X_full_train = train_df_merged.drop(columns=['Sales', 'Date', 'Open', 'Customers'])
    y_full_train = train_df_merged['Sales']  # Assuming 'Sales' is the target variable
    
    # Train the model on the prepared full training data
    best_xgb_model_full.fit(X_full_train, y_full_train)
    
    # Optionally, save the fully trained model
    joblib.dump(best_xgb_model_full, full_model_path)
    
    return best_xgb_model, best_xgb_model_full



#-----------------------------------------------------------------------------------##-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

import pandas as pd
import joblib

def predict_sales_for_test_df(test_df_merged, model_path='best_xgb_model_full_trained.pkl'):
    """
    Predicts sales for the test_df_merged dataset using the trained model.
    
    Parameters:
    - test_df_merged: The DataFrame containing the test data with pre-engineered features.
    - model_path: Path to the trained model.
    
    Returns:
    - result_df: DataFrame with the predicted sales and dates, sorted by Date.
    """

    # Load the trained model
    best_xgb_model_full = joblib.load(model_path)

    # Extract the feature names used during training
    training_features = best_xgb_model_full.get_booster().feature_names
    
    # Store the 'Date' column to add it back later
    dates = test_df_merged['Date'].copy()

    # Ensure the test_df_merged has the same features as the training set
    test_df_merged = test_df_merged.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Select only the columns needed for prediction
    test_df_merged = test_df_merged[training_features]

    # Predict sales for the test set
    test_df_merged['Predicted_Sales'] = best_xgb_model_full.predict(test_df_merged)
    
    # Add the 'Date' column back
    test_df_merged['Date'] = dates

    # Sort the DataFrame by 'Date' in ascending order
    test_df_merged = test_df_merged.sort_values(by='Date')

    return test_df_merged

#-----------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_sales_comparison_streamlit(train_df_merged, Test_df_future_sale):
    """
    Plots the comparison of actual sales for the last two months of the training data with predicted sales for Streamlit.

    Parameters:
    - train_df_merged (pd.DataFrame): DataFrame containing the actual sales data with a 'Date' and 'Sales' columns.
    - Test_df_future_sale (pd.DataFrame): DataFrame containing the predicted sales data with a 'Date' and 'Predicted_Sales' columns.

    Returns:
    - None: Displays a plot comparing the actual sales and predicted sales.
    """
    # Ensure that the 'Date' column is in datetime format
    train_df_merged['Date'] = pd.to_datetime(train_df_merged['Date'])

    # Filter the last two months of data from train_df_merged
    last_two_months_train = train_df_merged[train_df_merged['Date'] >= train_df_merged['Date'].max() - pd.DateOffset(months=2)]

    # Ensure the 'Date' column is in datetime format in the prediction DataFrame
    Test_df_future_sale['Date'] = pd.to_datetime(Test_df_future_sale['Date'])

    # Rename columns for clarity
    last_two_months_train = last_two_months_train[['Date', 'Sales']].rename(columns={'Sales': 'Actual_Sales'})
    Test_df_future_sale = Test_df_future_sale[['Date', 'Predicted_Sales']]

    # Combine the two datasets
    combined_sales = pd.concat([last_two_months_train, Test_df_future_sale])

    plt.figure(figsize=(14, 7))

    # Plot actual sales for the last two months
    plt.plot(last_two_months_train['Date'], last_two_months_train['Actual_Sales'], label='Actual Sales (Last 2 Months)', color='blue')

    # Plot predicted sales for the entire predicted period
    plt.plot(Test_df_future_sale['Date'], Test_df_future_sale['Predicted_Sales'], label='Predicted Sales', color='orange', linestyle='--')

    # Add titles and labels
    plt.title('Actual Sales (Last 2 Months) vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot in Streamlit
    st.pyplot(plt)




#-----------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

def plot_sales_comparison(train_df_merged, Test_df_future_sale):
    """
    Plots the comparison of actual sales for the last two months of the training data with predicted sales.

    Parameters:
    - train_df_merged (pd.DataFrame): DataFrame containing the actual sales data with a 'Date' and 'Sales' columns.
    - Test_df_future_sale (pd.DataFrame): DataFrame containing the predicted sales data with a 'Date' and 'Predicted_Sales' columns.

    Returns:
    - None: Displays a plot comparing the actual sales and predicted sales.
    """
    # Ensure that the 'Date' column is in datetime format
    train_df_merged['Date'] = pd.to_datetime(train_df_merged['Date'])

    # Filter the last two months of data from train_df_merged
    last_two_months_train = train_df_merged[train_df_merged['Date'] >= train_df_merged['Date'].max() - pd.DateOffset(months=2)]

    # Ensure the 'Date' column is in datetime format in the prediction DataFrame
    Test_df_future_sale['Date'] = pd.to_datetime(Test_df_future_sale['Date'])

    # Rename columns for clarity
    last_two_months_train = last_two_months_train[['Date', 'Sales']].rename(columns={'Sales': 'Actual_Sales'})
    Test_df_future_sale = Test_df_future_sale[['Date', 'Predicted_Sales']]

    # Combine the two datasets
    combined_sales = pd.concat([last_two_months_train, Test_df_future_sale])

    plt.figure(figsize=(14, 7))

    # Plot actual sales for the last two months
    plt.plot(last_two_months_train['Date'], last_two_months_train['Actual_Sales'], label='Actual Sales (Last 2 Months)', color='blue')

    # Plot predicted sales for the entire predicted period
    plt.plot(Test_df_future_sale['Date'], Test_df_future_sale['Predicted_Sales'], label='Predicted Sales', color='orange', linestyle='--')

    # Add titles and labels
    plt.title('Actual Sales (Last 2 Months) vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)

    # Show the plot
    plt.show()





