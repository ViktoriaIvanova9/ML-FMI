import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF, PCA
from scipy.sparse import csr_matrix

def main():    
    digits_df = pd.read_csv('../DATA/lcd-digits.csv', header=None).values

    plt.imshow(digits_df[0].reshape(13, 8), cmap='gray')
    plt.title('First image')
    plt.show()
    
    nmf = NMF(n_components=7).fit(digits_df)
    f1, ax1  = plt.subplots(2, 4, sharex=True, sharey=True)
    f1.suptitle('Features learned by NMF')
    ax1 = ax1.flatten()
    for i in range(7):
        ax1[i].imshow(nmf.components_[i].reshape(13, 8), cmap='gray')
    plt.show()

    pca = PCA(n_components=7).fit(digits_df)
    f2, ax2  = plt.subplots(2, 4, sharex=True, sharey=True)
    f2.suptitle('Features learned by PCA')
    ax2 = ax2.flatten()
    for i in range(7):
        ax2[i].imshow(pca.components_[i].reshape(13, 8), cmap='gray')
    plt.show()



if __name__ == '__main__':
    main()