import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)
    data = seeds_df.drop(columns=[0, 1, 2, 5, 6, 7], axis = 1)
    corr_coef = data.corr(method='pearson')

    plt.title(f'Pearson correlation: {round(np.mean(corr_coef), 2)}')
    plt.scatter(data[4], data[3]) # these are the width and length of the kernel but the correlation coeff is different
    plt.xlabel('Width of kernel')
    plt.ylabel('Length of kernel')
    plt.show()


if __name__ == '__main__':
    main()

    #  Is your hypothesis true? - yes, correlation is close to 1

