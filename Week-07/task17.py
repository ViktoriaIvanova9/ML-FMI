import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import root_mean_squared_error


def main():
    auto_df = pd.read_csv('../DATA/auto.csv')

    X = auto_df.drop(['origin', 'mpg'], axis=1)
    y = auto_df['mpg'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    dt_entropy = DecisionTreeRegressor(max_depth=8, random_state=3, min_samples_leaf=0.13).fit(X_train, y_train)
    # not sure that I understand what 13% of the data to be holded by a leaf
    y_pred_dt = dt_entropy.predict(X_test)

    linreg = LinearRegression().fit(X_train, y_train)
    y_pred_linreg = linreg.predict(X_test)

    print(f'Regression Tree test set RMSE: {round(root_mean_squared_error(y_test, y_pred_dt), 2)}')
    print(f'Linear Regression test set RMSE: {round(root_mean_squared_error(y_test, y_pred_linreg), 2)}')

if __name__ == '__main__':
    main()