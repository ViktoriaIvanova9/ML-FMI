import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix

def main():    
    wikipedia_df = pd.read_csv('../DATA/wikipedia-vectors.csv', index_col=0)

    titles = wikipedia_df.columns
    data = csr_matrix(wikipedia_df.T)

    nmf = NMF(n_components=6)
    weights = nmf.fit_transform(data)
    print(np.round(weights, 2)[:6])

    new_df = pd.DataFrame(nmf, index=titles, columns=np.arange(0, 6))
    selected_rows = new_df.loc[["Anne Hathaway", "Denzel Washington"]]
    print(selected_rows)

    vocabulary_df = pd.read_csv('../DATA/wikipedia-vocabulary-utf8.txt', header=None)
    print(f'The topic, that the articles about Anne Hathaway and Denzel Washington have in common, has the words:')
    print(pd.DataFrame(nmf.components_, columns=vocabulary_df).loc[3].nlargest())


if __name__ == '__main__':
    main()

    # Which feature has the highest value? - feature 3
    # What does this mean? - the topic of the article is connected with the third feature