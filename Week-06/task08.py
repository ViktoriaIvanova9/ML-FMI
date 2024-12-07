import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

def main():
    music_df = pd.read_csv('../DATA/music_clean.csv', index_col=0)

    X = music_df.drop(columns=['energy'])
    y = music_df['energy'].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear_reg_predict = LinearRegression().fit(X_train_scaled, y_train).predict(X_test_scaled)
    ridge_reg_predict = Ridge(alpha=0.1).fit(X_train_scaled, y_train).predict(X_test_scaled)

    print(f'Linear Regression Test Set RMSE: {root_mean_squared_error(y_test, linear_reg_predict)}')
    print(f'Ridge Test Set RMSE: {root_mean_squared_error(y_test, ridge_reg_predict)}')


if __name__ == '__main__':
    main()

