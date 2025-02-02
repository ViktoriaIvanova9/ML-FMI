import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

def main():    
    wikipedia_df = pd.read_csv('../DATA/wikipedia-vectors.csv', index_col=0)

    document_titles = wikipedia_df.columns
    idx_cristiano_ronaldo = document_titles.get_loc('Cristiano Ronaldo')

    nmf = NMF(n_components=6)
    weights = nmf.fit_transform(wikipedia_df.T)
    norm_features = normalize(nmf.components_)

    # new_df = pd.DataFrame(nmf, index=document_titles, columns=np.arange(0, 6))
    # cristiano_ronaldo = new_df.loc[["Cristiano Ronaldo"]]

    current_article = norm_features[:, idx_cristiano_ronaldo]
    similarities = norm_features.dot(current_article)
    print(similarities)

if __name__ == '__main__':
    main()