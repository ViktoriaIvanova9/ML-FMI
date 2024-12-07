import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    music_df = pd.read_csv('../DATA/music_clean.csv', index_col=0)

    X = music_df.drop('loudness', axis=1)
    y = music_df['loudness'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lasso_reg_score = Lasso(alpha=0.5).fit(X_train, y_train).score(X_test, y_test)

    print(music_df.iloc[:5]) # don't know why it prints different rows

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso_reg_scaled_score = Lasso(alpha=0.5).fit(X_train_scaled, y_train).score(X_test_scaled, y_test)

    print(f'Without scaling: {lasso_reg_score}')
    print(f'With scaling: {lasso_reg_scaled_score}')


if __name__ == '__main__':
    main()

    # Interpret the results - do the models perform well? - the R^2 result is better when the data is scaled. But these values are not very good.