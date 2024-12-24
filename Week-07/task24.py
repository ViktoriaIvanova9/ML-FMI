import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

def main():
    bikes_df = pd.read_csv('../DATA/bike_sharing.csv')

    bikes_df['datetime'] = pd.to_datetime(bikes_df['datetime'])

    bikes_df['year'] = bikes_df['datetime'].dt.year
    bikes_df['month'] = bikes_df['datetime'].dt.month
    bikes_df['day'] = bikes_df['datetime'].dt.day
    bikes_df['hour'] = bikes_df['datetime'].dt.hour
    bikes_df = bikes_df.drop(['datetime'], axis=1)

    X = bikes_df.drop(['count'], axis=1)
    y = bikes_df['count'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    steps = [('imputer', SimpleImputer(strategy='most_frequent')),
             ('scaler', StandardScaler())]
    pipeline = Pipeline(steps)

    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared =  pipeline.transform(X_test)

    gradient_boosting_regressor = GradientBoostingRegressor(max_depth=4, n_estimators=200, random_state=2).fit(X_train_prepared, y_train)
    # here possible one explanation won't be useless

    y_pred = gradient_boosting_regressor.predict(X_test_prepared)

    print(f'Test set RMSE: {round(root_mean_squared_error(y_pred, y_test), 2)}')

if __name__ == '__main__':
    main()