import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def main():
    sales_df = pd.read_csv('../DATA/advertising_and_sales_clean.csv')

    X = sales_df.drop(['sales', 'influencer'], axis=1)
    y = sales_df['sales'].values

    kf = KFold(n_splits=6, shuffle=True, random_state=5)

    linear_reg = LinearRegression()
    r2 = cross_val_score(linear_reg, X, y, cv=kf)

    print(f'Mean: {np.mean(r2)}')
    print(f'Standard Deviation: {np.std(r2)}')
    print(f'95% Confidence Interval: {np.quantile(r2, [0.025, 0.975])}')

    plt.plot(np.arange(1, 7), result)
    plt.title("R^2 per 6-fold split")
    plt.xlabel("# Split")
    plt.ylabel("R^2")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()