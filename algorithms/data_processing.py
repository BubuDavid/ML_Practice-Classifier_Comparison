import numpy as np
import pandas as pd

def preprocessing(X, y):
    X = pd.DataFrame(X, columns=['x1','x2'])
    y = np.array([np.array([i]) for i in y])
    return X, y

def postprocessing(X, y):
    try:
        X.drop('bias', inplace=True, axis=1)
    except:
        pass
    X = X.to_numpy()
    y = np.ravel(y)
    return X, y


if __name__ == '__main__':
    from read import read_files
    datasets = read_files('../')
    X, y = datasets[0]
    print(X.shape)
    print(y.shape)
    print(type(X))
    print(type(y))
    X, y = preprocessing(X, y)
    print("-----------------------------------")
    print(X.shape)
    print(y.shape)
    print(type(X))
    print(type(y))
    print("-----------------------------------")
    X, y = postprocessing(X, y)
    print(X.shape)
    print(y.shape)
    print(type(X))
    print(type(y))
    