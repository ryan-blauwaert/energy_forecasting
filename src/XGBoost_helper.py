import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def find_gridsearch_best_params(X_train, y_train, grid, tscv=5):
    """Uses sklearn's GridSearchCV to search for best combination
    of parameters from the input grid. 

    Args:
        X_train (arr): Training feature variable matrix
        y_train (arr): Training target variable matrix
        grid (dict): Dictionary of parameters to try in GridSearchCV;
        keys are XGBRegressor parameters; values are lists of 
        parameters to try
        tscv (int, optional): Number of TimeSeriesSplits to use in 
        cross validation. Defaults to 5.

    Returns:
        dict: Dictionary of best parameters based on cross validation
    """
    xgbr_gridsearch = GridSearchCV(XGBRegressor(), 
                                    grid, 
                                    cv=TimeSeriesSplit(n_splits=tscv),
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring = 'neg_root_mean_squared_error')
    
    xgbr_gridsearch.fit(X_train, y_train)
    return xgbr_gridsearch.best_params_

def fit_best_model(X_train, y_train, best_params):
    """Uses best parameters from GridSearchCV to instantiate best
    XGBRegressor; fits model using X_train and y_train arrays; generates
    predictions of target variable from X_test array.

    Args:
        X_train (arr): Training feature variable matrix
        y_train (arr): Training target variable matrix
        X_test (arr): Test feature variable matrix
        best_params (dict): Dictionary of best parameters for 
        XGBRegressor based on GridSearchCV
    """
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    return best_model

def predict_year_future(model, X_test):
    preds = model.predict(X_test)
    return preds

if __name__ == '__main__':

    pass