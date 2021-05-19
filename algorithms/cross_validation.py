import numpy as np
from algorithms.data_processing import preprocessing

def cross_val_mlp(clf, X, y, cv):
    #np.random.shuffle(X)
    bins = len(X) // (cv+1)
    score = []

    for i in range(cv):
        start = int(i*bins)
        end = int(i*bins+bins)
        train_x = X[start:end]
        train_y = y[start:end]
        try:
            test_x = np.concatenate(X[0:start],X[end:])
            test_y = np.concatenate(y[0:start],y[end:])
        except:
            test_x = X[end:]
            test_y = y[end:]
        
        if len(test_y) == 0:
            return score 

        train_x, train_y = preprocessing(train_x, train_y)
        test_x, test_y = preprocessing(test_x, test_y)

        clf.fit(train_x, train_y)
        predicted_y = clf.predict(test_x)
        predicted_y = np.round(predicted_y)
        
        suma = 0
        for i in range(len(predicted_y)):
            if predicted_y[i] == test_y[i]:
                suma += 1
        mean = suma / len(predicted_y)
        score.append(mean)
    
    return np.array(score)

if __name__ == '__main__':
    import time
    from read import read_files
    from perceptron_multicapa import FFNN
    from data_processing import preprocessing


    clf = FFNN(hidden_size=250, training_epochs=100)
    datasets = read_files()
    X, y = datasets[3]
    score = cross_val_mlp(clf, X, y, 10)
    print(score)