import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn import decomposition

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)

    data = seeds_df.drop(columns=[0, 1, 2, 5, 6, 7], axis = 1)

    pca = decomposition.PCA()
    pca_transformed = pca.fit_transform(data)
    pca_df = pd.DataFrame(pca_transformed, columns=['width', 'length'])
    corr_coef = pca_df.corr(method='pearson')
    plt.title(f'Pearson correlation: {round(np.mean(corr_coef), 2)}')
    plt.scatter(pca_df['width'], pca_df['length'])

    interval_list = np.arange(-0.8, 1, 0.2)
    plt.yticks(interval_list)

    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show() # the graphics looks the same but the correlation coefficient is not, don't know why


if __name__ == '__main__':
    main()

