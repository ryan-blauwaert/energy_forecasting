import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def find_gridsearch_best_params(X_train, y_train, grid, tscv=5):
    xgbr_gridsearch = GridSearchCV(XGBRegressor(), 
                                    grid, 
                                    cv=TimeSeriesSplit(n_splits=tscv),
                                    n_jobs=-1,
                                    verbose=True,
                                    scoring = 'neg_root_mean_squared_error')
    
    xgbr_gridsearch.fit(X_train, y_train)
    return xgbr_gridsearch.best_params

def fit_and_predict_best_model(X_train, y_train, X_test, best_params):
    best_model = XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    best_preds = best_model.predict(X_test)


if __name__ == '__main__':

    pass