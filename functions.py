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

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineering_and_split(train_df, test_df):
    """
    Performs feature engineering on train and test datasets, including date-related features,
    binary feature creation, lag features, moving averages, and cyclical transformations.
    Finally, the function aligns the test dataset with the train dataset's columns and displays
    the processed DataFrames.

    Args:
        train_df (pd.DataFrame): The training dataset.
        test_df (pd.DataFrame): The testing dataset.

    Returns:
        pd.DataFrame, pd.DataFrame: The processed training and testing DataFrames.
    """
    # Extract date-related features
    for df in [train_df, test_df]:
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week

    # Create binary features for holidays and promotions
    train_df['IsHoliday'] = train_df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)
    test_df['IsHoliday'] = test_df['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)

    train_df['IsPromo'] = train_df['Promo'].apply(lambda x: 1 if x == 1 else 0)
    test_df['IsPromo'] = test_df['Promo'].apply(lambda x: 1 if x == 1 else 0)

    # Calculate competition duration in months
    train_df['CompetitionOpenSince'] = 12 * (train_df['Year'] - train_df['CompetitionOpenSinceYear']) + (train_df['Month'] - train_df['CompetitionOpenSinceMonth'])
    test_df['CompetitionOpenSince'] = 12 * (test_df['Year'] - test_df['CompetitionOpenSinceYear']) + (test_df['Month'] - test_df['CompetitionOpenSinceMonth'])

    # Handle cases where the competition hasn't started yet (negative values)
    train_df['CompetitionOpenSince'] = train_df['CompetitionOpenSince'].apply(lambda x: max(x, 0))
    test_df['CompetitionOpenSince'] = test_df['CompetitionOpenSince'].apply(lambda x: max(x, 0))

    # Calculate Promo2 duration in weeks
    train_df['Promo2OpenSince'] = 52 * (train_df['Year'] - train_df['Promo2SinceYear']) + (train_df['WeekOfYear'] - train_df['Promo2SinceWeek'])
    test_df['Promo2OpenSince'] = 52 * (test_df['Year'] - test_df['Promo2SinceYear']) + (test_df['WeekOfYear'] - test_df['Promo2SinceWeek'])

    # Handle cases where the promo hasn't started yet (negative values)
    train_df['Promo2OpenSince'] = train_df['Promo2OpenSince'].apply(lambda x: max(x, 0))
    test_df['Promo2OpenSince'] = test_df['Promo2OpenSince'].apply(lambda x: max(x, 0))

    # Convert all values in 'StateHoliday' to strings and apply Label Encoding
    label_encoder = LabelEncoder()
    train_df['StateHoliday'] = label_encoder.fit_transform(train_df['StateHoliday'].astype(str))
    test_df['StateHoliday'] = label_encoder.transform(test_df['StateHoliday'].astype(str))

    # One-Hot Encoding for categorical variables: 'StoreType', 'Assortment', and 'PromoInterval'
    train_df = pd.get_dummies(train_df, columns=['StoreType', 'Assortment', 'PromoInterval'], drop_first=True)
    test_df = pd.get_dummies(test_df, columns=['StoreType', 'Assortment', 'PromoInterval'], drop_first=True)

    # Align columns between train and test sets (important if using different datasets)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Lag Features for Sales
    train_df['Sales_Lag_1'] = train_df['Sales'].shift(1)
    train_df['Sales_Lag_7'] = train_df['Sales'].shift(7)
    train_df['Sales_Lag_30'] = train_df['Sales'].shift(30)

    # Drop rows with NaN values resulting from lag features
    train_df.dropna(inplace=True)

    # Create lag features for 'Customers'
    train_df['Customers_Lag_1'] = train_df['Customers'].shift(1)
    train_df['Customers_Lag_7'] = train_df['Customers'].shift(7)
    train_df['Customers_Lag_30'] = train_df['Customers'].shift(30)

    # Apply the same for the test set
    test_df['Customers_Lag_1'] = test_df['Customers'].shift(1)
    test_df['Customers_Lag_7'] = test_df['Customers'].shift(7)
    test_df['Customers_Lag_30'] = test_df['Customers'].shift(30)

    # Fill NaN values with 0 or use forward fill/backward fill based on your data context
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    # Create lag features for 'Open'
    train_df['Open_Lag_1'] = train_df['Open'].shift(1)
    train_df['Open_Lag_7'] = train_df['Open'].shift(7)
    train_df['Open_Lag_30'] = train_df['Open'].shift(30)

    # Drop rows with NaN values created by the lag features
    train_df.dropna(inplace=True)

    # Create moving average features for 'Sales'
    train_df['Sales_MA_7'] = train_df['Sales'].rolling(window=7).mean()
    train_df['Sales_MA_30'] = train_df['Sales'].rolling(window=30).mean()

    # Optionally, create moving averages for 'Customers' if it was available
    train_df['Customers_MA_7'] = train_df['Customers'].rolling(window=7).mean()
    train_df['Customers_MA_30'] = train_df['Customers'].rolling(window=30).mean()

    # Drop NaN values caused by the rolling operation
    train_df.dropna(inplace=True)

    # Apply Sine and Cosine Transformations to cyclical features
    for df in [train_df, test_df]:
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['WeekOfYear_Sin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
        df['WeekOfYear_Cos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)

    # Drop any remaining NaN values
    train_df.dropna(inplace=True)
    test_df.fillna(0, inplace=True)

    # Display the processed DataFrames
    print("Processed Training DataFrame:")
    print(train_df.head())

    print("\nProcessed Testing DataFrame:")
    print(test_df.head())

    return train_df, test_df




#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#



def split_train_test(df, target_column='Sales', date_column='Date', cutoff_date='2015-06-01'):
    """
    Splits the given DataFrame into training and testing sets based on a cutoff date.
    
    Args:
        df (pd.DataFrame): The DataFrame to be split.
        target_column (str): The name of the target column. Defaults to 'Sales'.
        date_column (str): The name of the date column. Defaults to 'Date'.
        cutoff_date (str): The cutoff date for splitting the data. Defaults to '2015-06-01'.
    
    Returns:
        pd.DataFrame: X_train (features for training set)
        pd.Series: y_train (target for training set)
        pd.DataFrame: X_test (features for testing set)
        pd.Series: y_test (target for testing set)
    """
    # Ensure the date column is in datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Split the dataset into training and test sets based on the cutoff date
    train_set = df[df[date_column] < cutoff_date]
    test_set = df[df[date_column] >= cutoff_date]
    
    # Separate features and target variable for training and testing
    X_train = train_set.drop(columns=[target_column, date_column, 'Customers', 'Open'])
    y_train = train_set[target_column]
    
    X_test = test_set.drop(columns=[target_column, date_column, 'Customers', 'Open'])
    y_test = test_set[target_column]
    
    # Verify the split
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    print("Training set date range:", train_set[date_column].min(), "to", train_set[date_column].max())
    print("Test set date range:", test_set[date_column].min(), "to", test_set[date_column].max())
    
    return X_train, y_train, X_test, y_test

#-----------------------------------------------------------------------------------#
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


def hyperparameter_tuning_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Performs hyperparameter tuning using GridSearchCV, trains the best model, and evaluates it on the test set.

    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.

    Returns:
        dict: A dictionary containing the best model, predictions, and evaluation metrics (RMSE, MAE, R-squared).
    """
    # Define the XGBoost model
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
                               scoring='neg_root_mean_squared_error', cv=3, 
                               verbose=2, n_jobs=-1)

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

    # Print the evaluation metrics
    print(f"Best XGBoost Test RMSE: {test_rmse_best_xgb}")
    print(f"Best XGBoost Test MAE: {test_mae_best_xgb}")
    print(f"Best XGBoost Test R-squared: {test_r2_best_xgb}")

    # Return the best model and evaluation metrics
    return {
        'best_model': best_xgb_model,
        'predictions': y_test_pred_best_xgb,
        'metrics': {
            'rmse': test_rmse_best_xgb,
            'mae': test_mae_best_xgb,
            'r2': test_r2_best_xgb
        }
    }




#-----------------------------------------------------------------------------------


import pandas as pd

def create_model_results_dataframe(rf_metrics, xgb_metrics, xgb_tuned_metrics, lgb_metrics, best_xgb_params):
    """
    Creates a DataFrame summarizing the results of different models.

    Args:
        rf_metrics (dict): Evaluation metrics for the Random Forest model.
        xgb_metrics (dict): Evaluation metrics for the XGBoost model.
        xgb_tuned_metrics (dict): Evaluation metrics for the tuned XGBoost model.
        lgb_metrics (dict): Evaluation metrics for the LightGBM model.
        best_xgb_params (dict): Best hyperparameters found during XGBoost tuning.

    Returns:
        pd.DataFrame: A DataFrame summarizing the results of the models.
    """
    # Define the results for each model
    model_results = {
        'Model': ['Random Forest', 'XGBoost', 'XGBoost (Tuned)', 'LightGBM'],
        'RMSE': [rf_metrics['rmse'], xgb_metrics['rmse'], xgb_tuned_metrics['rmse'], lgb_metrics['rmse']],
        'MAE': [rf_metrics['mae'], xgb_metrics['mae'], xgb_tuned_metrics['mae'], lgb_metrics['mae']],
        'R-squared': [rf_metrics['r2'], xgb_metrics['r2'], xgb_tuned_metrics['r2'], lgb_metrics['r2']],
        'Hyperparameters': [
            'n_estimators=100, max_depth=10',  # Random Forest parameters
            'n_estimators=100, max_depth=10',  # XGBoost parameters
            str(best_xgb_params),              # Tuned XGBoost best parameters
            'n_estimators=100, max_depth=10'   # LightGBM parameters
        ]
    }

    # Create a DataFrame from the results
    results_df = pd.DataFrame(model_results)

    return results_df





#-----------------------------------------------------------------------------------







#-----------------------------------------------------------------------------------







#-----------------------------------------------------------------------------------






#-----------------------------------------------------------------------------------






