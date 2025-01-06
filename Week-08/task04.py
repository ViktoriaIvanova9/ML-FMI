import pandas as pd
from sklearn.cluster import KMeans

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)

    species_dict = {1 : 'Kama wheat', 
                    2 : 'Rosa wheat', 
                    3 : 'Canadian wheat'}
    seeds_df[7] = seeds_df[7].map(species_dict)

    data = seeds_df.drop(columns=7, axis = 1)
    target = seeds_df[7]

    kmeans = KMeans(n_clusters=3)
    labels = kmeans.fit_predict(data)

    df = pd.DataFrame({'labels': labels, 'species': target})
    print(pd.crosstab(df['labels'], df['species']))


if __name__ == '__main__':
    main()

