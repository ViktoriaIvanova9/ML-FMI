import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold

def main():    
    seeds_df = pd.read_csv("../DATA/seeds_dataset.txt", sep='\s+', header=None)

    data = seeds_df.drop(columns=7, axis = 1)
    target = seeds_df[7]

    model = manifold.TSNE(learning_rate=300)
    transformed = model.fit_transform(data)
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.title("t-SNE on grain dataset")
    plt.scatter(xs, ys, c=target)
    plt.show()


if __name__ == '__main__':
    main()

