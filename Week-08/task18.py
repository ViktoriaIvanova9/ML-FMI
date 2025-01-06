import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.sparse import csr_matrix

def main():    
    wikipedia_df = pd.read_csv('../DATA/wikipedia-vectors.csv', index_col=0)
    titles = wikipedia_df.columns

    data = wikipedia_df.T
    X = csr_matrix(data)

    steps = [('svd', TruncatedSVD(n_components=50)),
             ('kmeans', KMeans(n_clusters=6))]
    pipeline = Pipeline(steps)

    wiki_modeled = pipeline.fit_predict(X)
    wiki_modeled_sorted_indices = np.argsort(wiki_modeled)

    df = pd.DataFrame({'label': wiki_modeled[wiki_modeled_sorted_indices], 'article': np.array(titles)[wiki_modeled_sorted_indices.astype(int)]})
    print(df)

if __name__ == '__main__':
    main()