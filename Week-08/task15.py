import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn import decomposition

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)

    data = seeds_df.drop(columns=[0, 1, 2, 5, 6, 7], axis = 1)

    plt.title(f'First principle component')
    plt.scatter(data[4], data[3])
    plt.xlabel('Kernel Width')
    plt.ylabel('Kernel Length')

    pca = decomposition.PCA().fit(data)
    fpc = pca.components_[0]
    mean_data = np.mean(data, axis=0)
    plt.arrow(mean_data.iloc[1], mean_data.iloc[0], fpc[1], fpc[0], color='red')
    plt.show()


if __name__ == '__main__':
    main()

