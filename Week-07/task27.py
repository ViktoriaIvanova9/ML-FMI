import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

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

    param_grid = {
        'n_estimators' : [100, 350, 500],
        'max_features' : ['log2', 'auto', 'sqrt'],
        'min_samples_leaf' : [2, 10, 30]
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=2)
    dt = GridSearchCV(RandomForestRegressor(random_state=2), param_grid, scoring='neg_root_mean_squared_error', cv=kf).fit(X_train_prepared, y_train)

    print(f'Test set RMSE: {round(np.abs(dt), 2)}') # at this point I don't know, it is not returning a result

if __name__ == '__main__':
    main()