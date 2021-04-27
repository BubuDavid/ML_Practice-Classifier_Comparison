import numpy as np

def knn(X, y, z, k):
    """
    X Es un dataset de N instancias.
    Y son las etiquetas de clase.
    z es el vector de pruebas.
    k es el número de puntos a comparar.
    """
    # Calculating all the distances
    distances = distance_data_point(X, z)
    
    # Identifying the k nearest neighbors
    k_distances = []
    k_labels = []
    for _ in range(k):
        mini = min(distances)
        # Adding up to my k_distance list
        k_distances.append(mini)
        
        # Identifying the index for deleting
        pop = np.where(distances == mini)
        # Identifying the label
        k_labels.append(y[pop[0][0]])
        
        # Deleting things
        distances = np.delete(distances, pop[0][0])
        y = np.delete(y, pop[0][0])
        
    counting = {}
    for label in k_labels:
        try:
            counting[label] += 1
        except:
            counting[label] = 0
            
    maxi = 0
    label = -1
    for key in counting.keys():
        if counting[key] > maxi:
            maxi = counting[key]
            label = key
    
    return label

if __name__ == '__main__':
    print("Hola")