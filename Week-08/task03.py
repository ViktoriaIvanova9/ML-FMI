import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.cluster import KMeans

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)

    print(seeds_df)

    clusters = np.arange(1, 7)
    inertia_list = []

    for num_clusters in clusters:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(seeds_df)
        inertia_list.append(kmeans.inertia_)

    plt.plot(clusters, inertia_list)
    plt.scatter(clusters, inertia_list)
    plt.title('Inertia per number of clusters')
    plt.xlabel('number of clusters, k')
    plt.ylabel('Inertia')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # What's a good number of clusters in this case? Why?. - I think 3 because after 3 the inertia is not becoming signifficantly smaller

