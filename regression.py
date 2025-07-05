import pandas as pd
import numpy as np

def fetch_housing_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    df_raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    
    features = np.hstack([df_raw.values[::2, :], df_raw.values[1::2, :2]])
    target = df_raw.values[1::2, 2]

    columns = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    housing_df = pd.DataFrame(features, columns=columns)
    housing_df['PRICE'] = target
    return housing_df

housing_data = fetch_housing_data()
print("Dimensions:", housing_data.shape)
print(housing_data.head())