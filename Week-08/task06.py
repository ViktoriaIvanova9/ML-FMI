import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

def main():
    price_to_json = pd.read_json("../DATA/price_movements.txt")
    price_df = pd.DataFrame(price_to_json)

    print(f'Data shape: {np.shape(price_df)}')
    
    companies = ['Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo']

    steps = [('normalizer', Normalizer()),
             ('kmeans', KMeans(n_clusters=10))]
    pipeline = Pipeline(steps)
    price_modeled = pipeline.fit_predict(price_df)

    price_modeled_sorted_indices = np.argsort(price_modeled)
    
    df = pd.DataFrame({'labels': price_modeled[price_modeled_sorted_indices], 'species': np.array(companies)[price_modeled_sorted_indices.astype(int)]})
    # here again I don't understand why the result is changing
    print(df)

if __name__ == '__main__':
    main()

