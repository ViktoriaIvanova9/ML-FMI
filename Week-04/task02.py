import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  Ridge
from sklearn.model_selection import train_test_split

def main():
    sales_df = pd.read_csv('../DATA/advertising_and_sales_clean.csv')

    X = sales_df.drop(['sales', 'influencer'], axis=1)
    y = sales_df['sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    interval_array = np.linspace(0.99, 1.0, 10)
    alpha_values = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    ridge_dictionary = {}
    r2_results = []

    for curr_alpha in alpha_values:
        ridge_reg = Ridge(alpha=curr_alpha)
        ridge_reg.fit(X_train, y_train)
        r2 = ridge_reg.score(X_test, y_test)
        r2_results.append(r2)

        ridge_dictionary[curr_alpha] = r2

    print(f'Ridge scores per alpha: {ridge_dictionary}')

    plt.plot(alpha_values, r2_results)
    plt.title("R^2 per alpha")
    plt.xlabel("Alpha")
    plt.ylabel("R^2")
    plt.yticks(interval_array)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # "Do we have overfitting?" -  не
    # "Do we have underfitting?" - да
    # "How does heavy penalization affect model performance?" - R^2 е твърде голямо, 
    #                                       което значи че прекалено много “наказва“ силно отклонените стойности