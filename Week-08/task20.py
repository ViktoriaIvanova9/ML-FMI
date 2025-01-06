import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

def main():    
    digits_df = pd.read_csv('../DATA/lcd-digits.csv', index_col=0)
    
    plt.tight_layout()
    plt.show()
    



if __name__ == '__main__':
    main()

    # Which feature has the highest value? - feature 3
    # What does this mean? - the topic of the article is connected with the third feature