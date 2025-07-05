from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def train_optimized_models(X_train, y_train):
    trained_models = {}

    ridge_grid = {'alpha': [0.01, 0.1, 1, 10], 'solver': ['auto', 'svd', 'cholesky']}
    ridge_model = GridSearchCV(Ridge(), ridge_grid, cv=5)
    ridge_model.fit(X_train, y_train)
    trained_models['Ridge'] = ridge_model

    dt_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    dt_model = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_grid, cv=5)
    dt_model.fit(X_train, y_train)
    trained_models['DecisionTree'] = dt_model

    rf_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'max_features': ['sqrt', 'log2']}
    rf_model = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=5)
    rf_model.fit(X_train, y_train)
    trained_models['RandomForest'] = rf_model

    return trained_models