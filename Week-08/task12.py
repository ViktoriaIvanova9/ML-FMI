import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn import preprocessing
from sklearn import manifold

def main():
    price_to_json = pd.read_json("../DATA/price_movements.txt")
    price_df = pd.DataFrame(price_to_json)
    
    companies = ['Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo']

    price_df_normalized = preprocessing.normalize(price_df)
    model = manifold.TSNE(learning_rate=10)
    transformed = model.fit_transform(price_df_normalized)
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.title("t-SNE on the stock price dataset")
    plt.scatter(xs, ys)
    for i, _ in enumerate(companies):
        plt.annotate(companies[i], (transformed[i, 0], transformed[i, 1])) # don't understand why the result is changing
    plt.show()


if __name__ == '__main__':
    main()