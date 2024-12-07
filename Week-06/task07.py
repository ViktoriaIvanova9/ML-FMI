import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score

def main():
    music_df = pd.read_csv('../DATA/music_clean.csv', index_col=0)

    X = music_df.drop('energy', axis=1)
    y = music_df['energy'].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    linear_reg = LinearRegression()
    lin_reg_score = cross_val_score(linear_reg, X_train, y_train, cv=kf)
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg_score = cross_val_score(lasso_reg, X_train, y_train, cv=kf)    
    ridge_reg = Ridge(alpha=0.1)
    ridge_reg_score = cross_val_score(ridge_reg, X_train, y_train, cv=kf)

    plt.boxplot([lin_reg_score, ridge_reg_score, lasso_reg_score], tick_labels=['Linear regression', 'Ridge', 'Lasso'])
    plt.tight_layout()
    plt.show()

    # Does this mean that we are not using the test data at all?

if __name__ == '__main__':
    main()

    # Which model performs best? - Ridge and Linear