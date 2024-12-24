import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import root_mean_squared_error


def main():
    auto_df = pd.read_csv('../DATA/auto.csv')

    auto_dummies = pd.get_dummies(auto_df['origin'], drop_first=True, dtype=int)
    auto_dummies = pd.concat([auto_df, auto_dummies], axis=1)
    auto_dummies = auto_dummies.drop(columns=['origin'])

    X = auto_dummies.drop(['mpg'], axis=1)
    y = auto_dummies['mpg'].values

    print('Dataset X:')
    print(auto_dummies.head())
    print(auto_dummies.tail())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    kf = KFold(n_splits=10) # why no shuffling here??
    dt = DecisionTreeRegressor(max_depth=4, random_state=1, min_samples_leaf=0.26)
    dt_cv_score = cross_val_score(dt, X_train, y_train, scoring='neg_root_mean_squared_error', cv=kf)

    print(f'10-fold CV RMSE: {round(np.mean(np.abs(dt_cv_score)), 2)}')
    
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_train)
    print(f'Train RMSE: {round(root_mean_squared_error(y_train, y_pred), 2)}')

if __name__ == '__main__':
    main()

    # Does dt suffer from a high bias or a high variance problem? Why?. - RMSE is almost similar as in the train set, so I think no.