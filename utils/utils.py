import pandas as pd
import statsmodels.api as sm 

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error


def create_dummies(data, column_name):
    """
    Create dummy variables for a categorical column and append them to the original DataFrame.

    Parameters:
    - data: pandas DataFrame
    - column_name: str, the name of the categorical column for which to create dummy variables

    Returns:
    - data: pandas DataFrame, the updated DataFrame with dummy variables
    """
    
    dummies = pd.get_dummies(data[column_name], prefix=column_name)
    data = pd.concat([data, dummies], axis=1)

    return data


def split_step(data, data_type, by_subset=False, seed = 42, kernel = False):
    """
    Split the data into training and testing sets, create polynomial features, and calculate weights.

    Parameters:
    - data: pandas DataFrame, the input data
    - data_type: str, the type of data used for creating dummy variables
    - by_subset: bool, whether to split the data into subsets or use stratified split (default: False)
    - seed: int, random seed for reproducibility (default: 42)
    - kernel: bool, wbool, whether we are using Kernel Regression or not (default: False)

    Returns:
    - X_train_poly: pandas DataFrame, features for training set with polynomial features and dummy variables
    - X_test_poly: pandas DataFrame, features for testing set with polynomial features and dummy variables
    - X_train: pandas DataFrame, original features for training set
    - X_test: pandas DataFrame, original features for testing set
    - y_train: pandas Series, target variable for training set
    - y_test: pandas Series, target variable for testing set
    - train_weights: pandas Series, weights for training set
    - test_weights: pandas Series, weights for testing set
    """
    
    if data_type:
        data = create_dummies(data, data_type)
    
    if by_subset:
        X_train, X_test, y_train, y_test = train_test_split(data, (data['fuel_burn_total']), test_size=0.2, random_state=seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, (data['fuel_burn_total']), test_size=0.2, random_state=seed, stratify=data[['model']])

    train_weights = (X_train['depcount']*y_train)
    train_weights = train_weights/train_weights.sum()

    test_weights = (X_test['depcount']*y_test)
    test_weights = test_weights/test_weights.sum()

    X_train_poly = pd.DataFrame() 
    X_train_poly['dist'] = X_train['dist']
    X_train_poly['seats'] = X_train['seats']
    X_train_poly['inv_dist'] = 1/X_train['dist']


    X_test_poly = pd.DataFrame()
    X_test_poly['dist'] = X_test['dist']
    X_test_poly['seats'] = X_test['seats']
    X_test_poly['inv_dist'] = 1/X_test['dist']

    if not kernel:
        X_test_poly = sm.add_constant(X_test_poly)
        X_train_poly = sm.add_constant(X_train_poly)

    if data_type:
        X_train_poly = pd.concat([X_train_poly, data.loc[X_train.index, data.columns.str.startswith(f'{data_type}_')]], axis=1)
        X_test_poly = pd.concat([X_test_poly, data.loc[X_test.index, data.columns.str.startswith(f'{data_type}_')]], axis=1)
    
    return X_train_poly, X_test_poly, X_train, X_test, y_train, y_test, train_weights, test_weights


def data_subsets(data):
    """ Split the data into subsets based on the 'dist' column.

    Parameters:
    - data: pandas DataFrame, the input data

    Returns:
    - list of pandas DataFrames, subsets of the original data based on distance ranges
    """
     
    short = data[(data['dist'] < 500)]
    medium = data[(data['dist'] >= 500) & (data['dist'] < 1500)]
    medium_long = data[(data['dist'] >= 1500) & (data['dist'] < 4000)]
    long = data[data['dist'] > 4000]

    return [short, medium, medium_long, long]


def results_by_subset(model, X_test, data_type, kernel = False):
    """
    Print model evaluation results for different subsets of the test data.

    Parameters:
    - model: statsmodels OLS model object, the trained model
    - X_test: pandas DataFrame, features for the test set
    - data_type: str, the type of data used for creating dummy variables
    - kernel: bool, whether we are using Kernel Regression or not (default: False)

    Returns:
    - None
    """
     
    sets = data_subsets(X_test)
    for set_name, set_data in zip(['Short', 'Medium', 'Medium Long', 'Long'], sets):
        set_weights = (set_data["depcount"] * set_data["fuel_burn_total"])
        set_weights = set_weights / set_weights.sum()

        X_test_poly = pd.DataFrame()
        X_test_poly['dist'] = set_data['dist']
        X_test_poly['seats'] = set_data['seats']
        X_test_poly['inv_dist'] = 1 / set_data['dist']
        
        if not kernel:
            X_test_poly = sm.add_constant(X_test_poly)

        if data_type:
            X_test_poly = pd.concat([X_test_poly, set_data.loc[set_data.index, set_data.columns.str.startswith(f'{data_type}_')]], axis=1)

        print(f"\n{set_name} Set:")
        test_pred = model.predict(X_test_poly)
        print(f"Test Error, MAPE: {mean_absolute_percentage_error(set_data['fuel_burn_total'], test_pred, sample_weight= set_weights)*100}%")
    
