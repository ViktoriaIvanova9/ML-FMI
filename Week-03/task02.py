import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def main():
    sales_df = pd.read_csv('../DATA/advertising_and_sales_clean.csv')

    X = sales_df.drop(['sales', 'influencer'], axis=1)
    y = sales_df['sales'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lin_regression = LinearRegression()
    lin_regression.fit(X_train, y_train)
    predictions = lin_regression.predict(X_test)

    print(f'Predictions: {predictions[0:2]}')
    print(f'Actual Values: {y_test[0:2]}')

    r2 = lin_regression.score(X_test, y_test)
    rsme = np.sqrt((np.sum((y_test - predictions) ** 2))/len(predictions))

    print(f'R^2: {r2}')
    print(f'RMSE: {rsme}')

if __name__ == '__main__':
    main()