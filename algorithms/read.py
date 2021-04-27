import pandas as pd
import numpy as np

def read_files(extra=""):
    dirs = [extra + f'datasets\dataset_classifiers{i}.csv' for i in range(1,4)]
    data = []
    for dir in dirs:
        df = pd.read_csv(dir, sep=',', names=["x1", "x2", "y"])
        data.append([df.values[:, 0:2], df.values[:, 2]])
    
    return data


if __name__ == '__main__':
    data = read_files("../")
    print(len(data))