import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def retrieve_housing_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_data = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    features = np.hstack([raw_data.values[::2, :], raw_data.values[1::2, :2]])
    target = raw_data.values[1::2, 2]
    columns = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    housing_df = pd.DataFrame(features, columns=columns)
    housing_df['PRICE'] = target
    return housing_df

def partition_data(df):
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def assess_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2