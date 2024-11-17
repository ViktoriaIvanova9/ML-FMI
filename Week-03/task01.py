import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    sales_df = pd.read_csv('../DATA/advertising_and_sales_clean.csv')

    print(sales_df.iloc[0:2])

    f, (scatter1, scatter2, scatter3) = plt.subplots(1, 3, sharey=True, figsize=(10, 6))
    f.suptitle("Relationship between variables and target")
    scatter1.scatter(sales_df[['tv']], sales_df[['sales']])
    scatter1.set_xlabel('tv')
    scatter1.set_ylabel('sales')
    scatter2.scatter( sales_df[['radio']], sales_df[['sales']],)
    scatter2.set_xlabel('radio')
    scatter2.set_ylabel('sales')
    scatter3.scatter( sales_df[['social_media']], sales_df[['sales']])
    scatter3.set_xlabel('social_media')
    scatter3.set_ylabel('sales')

    plt.tight_layout()
    plt.show()

    print('Feature with highest correlation (from visual inspection): tv')

    lin_regression = LinearRegression()
    X_radio = sales_df[['radio']]
    y_sales = sales_df[['sales']]
    lin_regression.fit(X_radio, y_sales)

    predictions = lin_regression.predict(X_radio)
    print(f'First five predictions: {predictions[:5]}')

    plt.figure(figsize=(8, 6))
    plt.scatter(X_radio, y_sales)
    plt.title('Relationship between radio expenditures and sales')
    plt.plot(X_radio, predictions, 'r')
    plt.xlabel('Radio expenditure($)')
    plt.ylabel('Sales($)')
    plt.show()

if __name__ == '__main__':
    main()