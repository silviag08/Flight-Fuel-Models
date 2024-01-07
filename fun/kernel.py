import pandas as pd
import statsmodels.api as sm 

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.kernel_ridge import KernelRidge


from utils.utils import split_step, results_by_subset, data_subsets


def kernel_fit(X_train, y_train, gamma=0.005):
    """
    Fit a Kernel Ridge Regression model using Laplacian kernel.

    Parameters:
    - X_train: pandas DataFrame, features for the training set
    - y_train: pandas Series, target variable for the training set
    - gamma: float, regularization parameter for the Laplacian kernel (default: 0.005)

    Returns:
    - krr: KernelRidge model object
    """
    krr = KernelRidge(kernel="laplacian", gamma=gamma).fit(X_train, y_train)
    return krr


def evaluation(X_train, X_test, y_train, y_test, model, train_weights, test_weights):
    """
    Evaluate a model using Mean Absolute Percentage Error (MAPE) on training and testing sets.

    Parameters:
    - X_train: pandas DataFrame, features for the training set
    - X_test: pandas DataFrame, features for the testing set
    - y_train: pandas Series, target variable for the training set
    - y_test: pandas Series, target variable for the testing set
    - model: trained model
    - train_weights: pandas Series, weights for the training set
    - test_weights: pandas Series, weights for the testing set+
    """
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    print(f"Train Error, MAPE: {mean_absolute_percentage_error(y_train, train_pred, sample_weight=train_weights)*100}%")
    print(f"Test Error, MAPE: {mean_absolute_percentage_error(y_test, test_pred, sample_weight=test_weights)*100}%")




def run_kernel_model(data_path, data_type=None, subset=True, kernel = False):
    """
    Run a kernel regression model on the specified dataset.

    Parameters:
    - data_path: str, path of the data
    - data_type: str, feature used for creating dummy variables (default: None)
    - subset: bool, whether to evaluate results by subsets (default: True)
    - kernel: bool, whether to use a kernel (default: False)

    Returns:
    - model: trained kernel model
    """

    data = pd.read_csv(data_path)
    X_train_poly, X_test_poly, X_train, X_test, y_train, y_test, train_weights, test_weights = split_step(data, data_type, by_subset=False ,seed=42 ,kernel=kernel)
    model = kernel_fit(X_train_poly, y_train)

    evaluation(X_train_poly, X_test_poly, y_train, y_test, model, train_weights, test_weights)

    if subset:
        results_by_subset(model, X_test, data_type, kernel)
        
    return model


def run_kernel_model_by_subset(data_path,data_type=None, kernel= False):
    """
    Run a kernel model on different subsets of the dataset.

    Parameters:
    - data_type: str, the type of data used for creating dummy variables (default: None)
    - kernel: bool, whether to use a kernel (default: False)
    """
    data = pd.read_csv(data_path)
    sets = data_subsets(data)

    for set_name, set_data in zip(['Short', 'Medium', 'Medium Long', 'Long'], sets):
        X_train_poly, X_test_poly, X_train, X_test, y_train, y_test, train_weights, test_weights = split_step(set_data, data_type, by_subset=True, kernel=kernel)
        
        model = kernel_fit(X_train_poly, y_train)
        evaluation(X_train_poly, X_test_poly, y_train, y_test, model, train_weights, test_weights)