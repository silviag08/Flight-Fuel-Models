import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

from utils.utils import split_step, results_by_subset, data_subsets

def run_polynomial_model(data_path,data_type=None, subset=True):
    """
    Run a polynomial model on the specified dataset.

    Parameters:
    - data_type: str, feature used for creating dummy variables (default: None)
    - subset: bool, whether to evaluate results by subsets (default: True)
    """

    data = pd.read_csv(data_path)

    X_train_poly, X_test_poly, X_train, X_test, y_train, y_test, train_weights, test_weights = split_step(data, data_type)
    model = sm.OLS(y_train, X_train_poly).fit()

    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)

    print(f"Train Error, MAPE: {mean_absolute_percentage_error(y_train, train_pred, sample_weight=train_weights)*100}%")
    print(f"Test Error, MAPE: {mean_absolute_percentage_error(y_test, test_pred, sample_weight=test_weights)*100}%")

    if subset:
        results_by_subset(model, X_test, data_type)


def run_polynomial_model_by_subset(data_path,data_type=None):
    """
    Run a polynomial model on different subsets of the dataset.

    Parameters:
    - data_path: str, path of the data
    - data_type: str, feature used for creating dummy variables (default: None)
    """
    data = pd.read_csv(data_path)
    sets = data_subsets(data)

    for set_name, set_data in zip(['Short', 'Medium', 'Medium Long', 'Long'], sets):
        X_train_poly, X_test_poly, X_train, X_test, y_train, y_test, train_weights, test_weights = split_step(set_data, data_type, by_subset=True)
        model = sm.OLS(y_train, X_train_poly).fit()

        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)
        
        print(f"\nModel training only with {set_name} Set:")
        print(f"Train Error, MAPE: {mean_absolute_percentage_error(y_train, train_pred, sample_weight=train_weights)*100}%")
        print(f"Test Error, MAPE: {mean_absolute_percentage_error(y_test, test_pred, sample_weight=test_weights)*100}%")