import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from sklearn import preprocessing

def main():    
    eurovision_df = pd.read_csv("../DATA/eurovision_voting.csv")
    
    data = eurovision_df.drop(eurovision_df.columns[0], axis=1)
    list_countries = eurovision_df.iloc[:, 0].to_list()

    mergings = hierarchy.linkage(data, method='single')
    hierarchy.dendrogram(mergings, labels=list_countries, leaf_font_size=6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

