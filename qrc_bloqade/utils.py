import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_split(X, y, test_size=0.4, random_state=402):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,shuffle=True, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test
def normalize(X, y):
    scalar_x= MinMaxScaler(
    feature_range=(-1,1)
    )
    X_norm = scalar_x.fit_transform(X)
    scalar_y= MinMaxScaler(
    feature_range=(0,1)
    )
    y_norm = scalar_y.fit_transform(y.reshape(-1,1)).flatten()
    return X_norm, y_norm, scalar_x, scalar_y
 
def get_predictor():
    return RandomForestRegressor(n_estimators=50, random_state=402)

def print_readout(readout_indices, all_readout):
    selected= [all_readout[i] for i in readout_indices]
    print("readout selected")
    for s in selected:
        print(f" {s}")