import  numpy as np

def my_sign_function(value):
    if type(value) == float or type(value) == np.float64:
        if value > 0:
            return 1
        return 0
    
    for index, v in enumerate(value):
        if v > 0:
            value[index] = 1
        else:
            value[index] = 0
    return value

class Perceptron:
    def __init__(self, W=np.random.rand(2), l_rate=0.01, b=np.random.rand()):
        self.W = W
        self.b = b
        self.l_rate = l_rate

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            for inputs, label in zip(X, y):
                predict = self.predict(inputs)
                self.W = self.W + self.l_rate*(label - predict)*inputs
                self.b = self.b + self.l_rate*(label - predict)

    def predict(self, X, act_funct=my_sign_function):
        product = np.dot(X, self.W) + self.b
        out = act_funct(product)
        return out
    
    def get_params(self, deep=True):
        dict = {'W':self.W, 'l_rate':self.l_rate, 'b':self.b}
        return dict

if __name__ == '__main__':
    from read import read_files

    # Reading files
    dataset = read_files("../")
    # Trying with just one dataset    
    X, y = dataset[0]
    # Defining our perceptron
    percy = Perceptron()
    # Fitting our preceptron
    percy.fit(X, y)
    print(percy.b, percy.W)


    print(type(10)== int)
