import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

def main():
    sales_df = pd.read_csv('../DATA/advertising_and_sales_clean.csv')

    X = sales_df.drop(['sales', 'influencer'], axis=1)
    y = sales_df['sales'].values

    feature_list = ["tv", "radio", "social_media"]

    lasso_reg = Lasso(alpha=0.1)
    coeff_list = lasso_reg.fit(X, y).coef_

    print(f'Lasso coefficients per feature: {dict(zip(feature_list, np.round(coeff_list, 4)))}')

    plt.bar(feature_list, coeff_list)
    plt.title("Feature importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    #  "Which is the most important to predict sales?" - tv