import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MaxAbsScaler, Normalizer

def main():    
    artists_df = pd.read_csv('../DATA/artists.csv', header=None)
    listeners_df = pd.read_csv('../DATA/scrobbler-small-sample.csv')

    rows = listeners_df['artist_offset']
    columns = listeners_df['user_offset']
    data = listeners_df['playcount']

    sparse_matrix = csr_matrix((data, (rows, columns)))

    pipeline = Pipeline([('norm', Normalizer()),
                        ('scaler', MaxAbsScaler()),
                        ('nmf', NMF(n_components=20))])

    pipeline.fit(sparse_matrix)

    print(sparse_matrix.toarray())

if __name__ == '__main__':
    main()

    #  If you were a big fan of Bruce Springsteen which other musical artists might you like?
    #    Leonard Cohen
    #    Neil Young  
    #    The Beach Boys 
    #    Van Morrison 