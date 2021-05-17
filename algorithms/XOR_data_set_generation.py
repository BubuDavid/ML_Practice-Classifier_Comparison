import numpy as np
import pandas as pd

def XOR_dataset_creator(seed=True):
    if seed:       
        # initiating random number
        np.random.seed(11)

    #### Creating the dataset
    # mean and standard deviation for the x belonging to the first class
    mu_x1, sigma_x1 = 0, 0.1

    # creating the first distribution
    d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                    'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                    'type': 0})

    d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                    'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                    'type': 1})

    d3 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                    'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                    'type': 0})

    d4 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                    'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                    'type': 1})

    data = pd.concat([d1, d2, d3, d4], ignore_index=True)
    return data

if __name__ == '__main__':
    import seaborn as sns
    sns.set()

    data = XOR_dataset_creator()
    data.to_csv('../datasets/dataset_classifiers4.csv',
    header = False, index =  False)
    print(data)

    #Plotting 
    #ax = sns.scatterplot(x="x1", y="x2", hue="type",
    #                  data=data)

    #ax.figure.savefig('out.png')