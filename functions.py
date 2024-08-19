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
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

    # Apply log transformation
    df[column] = np.log1p(df[column])

    # Plot the distribution after log transformation
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of Log-Transformed {column}')
    plt.show()

    return df
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


import seaborn as sns
import matplotlib.pyplot as plt

def univariate_eda(df):
    """
    Performs univariate exploratory data analysis (EDA) on the given DataFrame.
    
    The function plots the distributions of Sales, Customers, StoreType, and CompetitionDistance.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
    
    Returns:
        None: The function outputs the plots directly.
    """
    # Set up the figure and axis for subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

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

    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

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
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Promo', y='Sales', data=df)
    plt.title('Sales vs Promo')
    plt.show()

    # Sales vs StoreType
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='StoreType', y='Sales', data=df)
    plt.title('Sales vs StoreType')
    plt.show()

    # Sales vs CompetitionDistance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='CompetitionDistance', y='Sales', data=df)
    plt.title('Sales vs Competition Distance')
    plt.show()

#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

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
    plt.show()

    # Sales by DayOfWeek
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='DayOfWeek', y=sales_col, data=df)
    plt.title('Sales by Day of the Week')
    plt.show()

    # Sales by Month
    df['Month'] = df[date_col].dt.month
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Month', y=sales_col, data=df)
    plt.title('Sales by Month')
    plt.show()



#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


def correlation_heatmap(df, target_col='Sales'):
    """
    Generates and displays a correlation heatmap for numerical features in the DataFrame.
    Additionally, it prints the correlations of all features with the target column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        target_col (str): The name of the target column to display correlations against. Defaults to 'Sales'.
    
    Returns:
        pd.Series: Correlation values of all features with the target column.
    """
    # Select only the numeric columns
    numeric_columns = df.select_dtypes(include=['number'])

    # Calculate the correlation matrix for the numerical features
    corr_matrix = numeric_columns.corr()

    # Plot the heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

    # Display the correlation matrix to identify strong correlations with the target column
    corr_target = corr_matrix[target_col].sort_values(ascending=False)
    print(corr_target)
    
    return corr_target


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

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
    print(sales_by_promo)

    # Plotting the sales by promotion status
    plt.figure(figsize=(4, 3))
    sales_by_promo.plot(kind='bar', color='orange')
    plt.xlabel('Promotion Status (0 = No, 1 = Yes)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales with and without Promotions')
    plt.xticks(rotation=0)
    plt.show()







#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

def plot_sales_by_competition_distance(df, bins=10):
    """
    This function takes a DataFrame and plots the average sales against competition distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing sales data, with 'CompetitionDistance' and 'Sales' columns.
    bins (int): Number of bins to divide the competition distance into. Default is 10.

    Returns:
    None: Displays a line chart of average sales by competition distance.
    """
    # Analyze sales against competition distance
    sales_by_competition_distance = df.groupby(pd.cut(df['CompetitionDistance'], bins=bins))['Sales'].mean()
    print(sales_by_competition_distance)

    # Plotting the sales by competition distance
    plt.figure(figsize=(6, 3))
    sales_by_competition_distance.plot(kind='line', marker='o', color='green')
    plt.xlabel('Competition Distance (binned)')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Competition Distance')
    plt.xticks(rotation=45)
    plt.show()




#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#



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
    plt.show()

#


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#

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
    plt.show()

    # Calculate average sales by assortment
    sales_by_assortment = df.groupby('Assortment')['Sales'].mean()

    # Plotting the average sales by assortment
    plt.figure(figsize=(8, 6))
    sales_by_assortment.plot(kind='bar', color='coral')
    plt.xlabel('Assortment Type')
    plt.ylabel('Average Sales')
    plt.title('Average Sales by Assortment Type')
    plt.xticks(rotation=0)
    plt.show()





#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


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
    plt.show()

    # Applying K-means clustering with the chosen number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Add the cluster labels to the original store data
    store_clustering_data['Cluster'] = clusters

    # Analyze the characteristics of each cluster by calculating the mean of each feature within clusters
    cluster_analysis = store_clustering_data.groupby('Cluster').mean()

    # Display the cluster analysis
    print(cluster_analysis)

    # Plotting the cluster centers for visualization
    plt.figure(figsize=(12, 6))
    cluster_analysis[['Sales', 'Customers', 'Promo', 'CompetitionDistance']].plot(kind='bar')
    plt.title('Cluster Analysis: Sales, Customers, Promo, and Competition Distance by Cluster')
    plt.xticks(rotation=0)
    plt.show()

    return store_clustering_data





#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#



def feature_engineering(train_df_merged, test_df_merged):
    # Extract date-related features
    for df in [train_df_merged, test_df_merged]:
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Create binary features for holidays and promotions
    for df in [train_df_merged, test_df_merged]:
        df['IsHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
        df['IsPromo'] = df['Promo'].apply(lambda x: 1 if x == 1 else 0)

    # Calculate competition duration in months
    for df in [train_df_merged, test_df_merged]:
        df['CompetitionOpenSince'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['CompetitionOpenSince'] = df['CompetitionOpenSince'].apply(lambda x: max(x, 0))

    # Calculate Promo2 duration in weeks
    for df in [train_df_merged, test_df_merged]:
        df['Promo2OpenSince'] = 52 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek'])
        df['Promo2OpenSince'] = df['Promo2OpenSince'].apply(lambda x: max(x, 0))

    # Convert 'StateHoliday' to strings and apply Label Encoding
    label_encoder = LabelEncoder()
    for df in [train_df_merged, test_df_merged]:
        df['StateHoliday'] = df['StateHoliday'].astype(str)
    train_df_merged['StateHoliday'] = label_encoder.fit_transform(train_df_merged['StateHoliday'])
    test_df_merged['StateHoliday'] = label_encoder.transform(test_df_merged['StateHoliday'])

    # One-Hot Encoding for categorical variables
    train_df_merged = pd.get_dummies(train_df_merged, columns=['StoreType', 'Assortment', 'PromoInterval'], drop_first=True)
    test_df_merged = pd.get_dummies(test_df_merged, columns=['StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

    # Align columns between train and test sets
    test_df_merged = test_df_merged.reindex(columns=train_df_merged.columns, fill_value=0)

    # Create lag features
    for lag in [1, 7, 30]:
        train_df_merged[f'Customers_Lag_{lag}'] = train_df_merged['Customers'].shift(lag)
        test_df_merged[f'Customers_Lag_{lag}'] = test_df_merged['Customers'].shift(lag)
        train_df_merged[f'Open_Lag_{lag}'] = train_df_merged['Open'].shift(lag)
        test_df_merged[f'Open_Lag_{lag}'] = test_df_merged['Open'].shift(lag)
    
    # Fill NaN values
    train_df_merged.fillna(0, inplace=True)
    test_df_merged.fillna(0, inplace=True)

    # Create moving average features for 'Sales' and 'Customers'
    for window in [7, 30]:
        train_df_merged[f'Sales_MA_{window}'] = train_df_merged['Sales'].rolling(window=window).mean()
        test_df_merged[f'Sales_MA_{window}'] = train_df_merged['Sales'].rolling(window=window).mean()
        train_df_merged[f'Customers_MA_{window}'] = train_df_merged['Customers'].rolling(window=window).mean()
        test_df_merged[f'Customers_MA_{window}'] = train_df_merged['Customers'].rolling(window=window).mean()
    
    # Drop NaN values caused by the rolling operation in the train set
    train_df_merged.dropna(inplace=True)

    # Apply Sine and Cosine Transformations to cyclical features
    for df in [train_df_merged, test_df_merged]:
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)

    return train_df_merged, test_df_merged

#-----------------------------------------------------------------------------------#
def split_and_separate_features(train_df_merged, cutoff_date='2015-06-01'):
    # Split the dataset into training and test sets based on the cutoff date
    train_set = train_df_merged[train_df_merged['Date'] < cutoff_date]
    test_set = train_df_merged[train_df_merged['Date'] >= cutoff_date]

    # Separate features and target variable for training and testing
    X_train = train_set.drop(columns=['Sales', 'Date', 'Customers', 'Open'])
    y_train = train_set['Sales']

    X_test = test_set.drop(columns=['Sales', 'Date', 'Open', 'Customers'])
    y_test = test_set['Sales']

    # Print the split verification
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("Training set date range:", train_set['Date'].min(), "to", train_set['Date'].max())
    print("Test set date range:", test_set['Date'].min(), "to", test_set['Date'].max())

    return X_train, y_train, X_test, y_test


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


def train_and_evaluate_lightgbm(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10, random_state=42):
    """
    Converts data types, cleans feature names, trains a LightGBM model, and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.
        n_estimators (int): Number of boosting rounds. Defaults to 100.
        max_depth (int): Maximum depth of the trees. Defaults to 10.
        random_state (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: A dictionary containing the model, predictions, and evaluation metrics (RMSE, MAE, R-squared).
    """
    # Convert specific columns to float
    columns_to_convert = ['WeekOfYear', 'WeekOfYear_Sin', 'WeekOfYear_Cos']
    for col in columns_to_convert:
        X_train[col] = X_train[col].astype(float)
        X_test[col] = X_test[col].astype(float)
    
    # Clean the feature names by replacing special characters with underscores
    X_train.columns = X_train.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    X_test.columns = X_test.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

    # Verify the cleaned feature names
    print("Cleaned feature names in X_train:", X_train.columns)

    # Check the data types of the specific columns
    print("Data types of specific columns in X_train:", X_train[columns_to_convert].dtypes)

    # Initialize the LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Train the model on the training data
    lgb_model.fit(X_train, y_train)

    # Predict on the test set
    y_test_pred_lgb = lgb_model.predict(X_test)

    # Evaluate model performance on the test set
    test_rmse_lgb = mean_squared_error(y_test, y_test_pred_lgb, squared=False)
    test_mae_lgb = mean_absolute_error(y_test, y_test_pred_lgb)
    test_r2_lgb = r2_score(y_test, y_test_pred_lgb)

    # Print the evaluation metrics
    print(f"LightGBM Test RMSE: {test_rmse_lgb}")
    print(f"LightGBM Test MAE: {test_mae_lgb}")
    print(f"LightGBM Test R-squared: {test_r2_lgb}")

    # Return the model and evaluation metrics
    return {
        'model': lgb_model,
        'predictions': y_test_pred_lgb,
        'metrics': {
            'rmse': test_rmse_lgb,
            'mae': test_mae_lgb,
            'r2': test_r2_lgb
        }
    }




#-----------------------------------------------------------------------------------


from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def tune_and_evaluate_xgboost(X_train, y_train, X_test, y_test):
    # Define the model
    xgb_model = xgb.XGBRegressor(random_state=42, enable_categorical=True)

    # Define the grid of hyperparameters
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
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

    return best_xgb_model, test_rmse_best_xgb, test_mae_best_xgb, test_r2_best_xgb




#-----------------------------------------------------------------------------------



def create_display_and_visualize_model_results(grid_search, test_rmse, test_mae, test_r2):
    """
    Creates a DataFrame summarizing the results of different models, displays it, and visualizes the results.
    
    Parameters:
    - grid_search: The GridSearchCV object used to tune the XGBoost model.
    - test_rmse: RMSE of the tuned XGBoost model on the test set.
    - test_mae: MAE of the tuned XGBoost model on the test set.
    - test_r2: R-squared of the tuned XGBoost model on the test set.
    
    Returns:
    - results_df: A DataFrame summarizing the performance and hyperparameters of each model.
    """
    
    # Define the results for each model
    model_results = {
        'Model': ['Random Forest', 'XGBoost', 'XGBoost (Tuned)', 'LightGBM'],
        'RMSE': [0.4511, 0.2893, test_rmse, 0.4372],
        'MAE': [0.2217, 0.0751, test_mae, 0.1960],
        'R-squared': [0.9786, 0.9912, test_r2, 0.9799],
        'Hyperparameters': [
            'n_estimators=100, max_depth=10',  # Random Forest parameters
            'n_estimators=100, max_depth=10',  # XGBoost parameters
            str(grid_search.best_params_),     # Tuned XGBoost best parameters
            'n_estimators=100, max_depth=10'   # LightGBM parameters
        ]
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame(model_results)

    # Display the DataFrame
    print(results_df)

    # Visualize the RMSE of each model
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Model'], results_df['RMSE'], color='skyblue')
    plt.xlabel('RMSE')
    plt.title('Model RMSE Comparison')
    plt.show()

    # Visualize the MAE of each model
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Model'], results_df['MAE'], color='lightcoral')
    plt.xlabel('MAE')
    plt.title('Model MAE Comparison')
    plt.show()

    # Visualize the R-squared of each model
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Model'], results_df['R-squared'], color='lightgreen')
    plt.xlabel('R-squared')
    plt.title('Model R-squared Comparison')
    plt.show()

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



#-----------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#
def predict_future_sales(test_df_merged, model_path='best_xgb_model_full_trained.pkl', output_csv='predicted_sales_test_df_merged.csv'):
    """
    Aligns the test dataset with the training dataset columns, predicts future sales, 
    and saves the predictions to a CSV file.
    
    Parameters:
    - test_df_merged: The test DataFrame that needs to be aligned and predicted.
    - model_path: Path to the trained model to be used for prediction.
    - output_csv: Path to the output CSV file where predictions will be saved.
    
    Returns:
    - test_df_merged: The test DataFrame with the predicted sales added.
    """
    
    # Define the common columns to ensure alignment between train and test sets
    common_columns = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                      'CompetitionDistance', 'CompetitionOpenSinceMonth', 
                      'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 
                      'Promo2SinceYear', 'Month', 'Year', 'WeekOfYear', 
                      'IsHoliday', 'IsPromo', 'CompetitionOpenSince', 
                      'Promo2OpenSince', 'StoreType_b', 'StoreType_c', 
                      'StoreType_d', 'Assortment_b', 'Assortment_c', 
                      'PromoInterval_Jan,Apr,Jul,Oct', 'PromoInterval_Mar,Jun,Sept,Dec', 
                      'PromoInterval_None', 'Sales_Lag_1', 'Sales_Lag_7', 
                      'Sales_Lag_30', 'Customers_Lag_1', 'Customers_Lag_7', 
                      'Customers_Lag_30', 'Open_Lag_1', 'Open_Lag_7', 
                      'Open_Lag_30', 'Sales_MA_7', 'Sales_MA_30', 
                      'Customers_MA_7', 'Customers_MA_30', 'DayOfWeek_Sin', 
                      'DayOfWeek_Cos', 'Month_Sin', 'Month_Cos', 
                      'WeekOfYear_Sin', 'WeekOfYear_Cos']

    # Identify missing columns in test_df_merged
    missing_columns = [col for col in common_columns if col not in test_df_merged.columns]
    if missing_columns:
        print("Missing columns in test_df_merged:", missing_columns)
        # Add the missing columns with default values (0)
        for col in missing_columns:
            test_df_merged[col] = 0

    # Reorder the test_df_merged columns to match the training data order
    X_test_full = test_df_merged[common_columns]

    # Load the trained model
    best_xgb_model_full = joblib.load(model_path)

    # Predict sales for the entire timeframe of the test set
    predicted_sales = best_xgb_model_full.predict(X_test_full)

    # Optionally, add the predictions to the test_df_merged DataFrame
    test_df_merged['Predicted_Sales'] = predicted_sales

    # Save the predictions to a CSV file
    test_df_merged.to_csv(output_csv, index=False)

    return test_df_merged


#-----------------------------------------------------------------------------------




def plot_sales_history_and_predictions(train_df_merged, test_df_merged):
    """
    Plots the historical sales for the last two months and future predicted sales.

    Parameters:
    - train_df_merged: The training DataFrame with historical sales data.
    - test_df_merged: The test DataFrame with predicted sales data.

    The function will display a plot with two subplots: one for the last two months of historical sales and one for future predicted sales.
    """

    # Ensure 'Date' is in datetime format
    train_df_merged['Date'] = pd.to_datetime(train_df_merged['Date'])
    test_df_merged['Date'] = pd.to_datetime(test_df_merged['Date'])

    # Filter the last two months of historical data
    last_two_months = train_df_merged[train_df_merged['Date'] >= train_df_merged['Date'].max() - pd.DateOffset(months=2)]

    # Prepare the daily sales for historical data
    historical_sales = last_two_months.groupby('Date')['Sales'].sum().reset_index()

    # Prepare the daily sales for predicted data
    future_predictions = test_df_merged.groupby('Date')['Predicted_Sales'].sum().reset_index()

    # Create the plot with two subplots
    plt.figure(figsize=(14, 6))

    # Plot for the last two months of historical data
    plt.subplot(1, 2, 1)
    plt.plot(historical_sales['Date'], historical_sales['Sales'], color='blue', label='Historical Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Historical Sales (Last Two Months)')
    plt.xticks(rotation=45)
    plt.legend()

    # Plot for the future predictions
    plt.subplot(1, 2, 2)
    plt.plot(future_predictions['Date'], future_predictions['Predicted_Sales'], color='orange', label='Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales')
    plt.title('Future Predicted Sales')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()



#-----------------------------------------------------------------------------------






#-----------------------------------------------------------------------------------






