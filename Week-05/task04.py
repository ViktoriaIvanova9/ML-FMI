import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df.drop('glucose', axis=1)
    y = diabetes_df['glucose'].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    interval_array = np.linspace(0.00001, 1.0, 20)
    param_grid = {
    'alpha': interval_array
    }

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    lasso_reg = GridSearchCV(Lasso(), param_grid, cv=kf).fit(X_train, y_train)

    print(f'Tuned lasso paramaters: {lasso_reg.best_params_}')
    print(f'Tuned lasso score: {lasso_reg.best_score_}')


if __name__ == '__main__':
    main()

    # Does using optimal hyperparameters guarantee a high performing model? - No, here R^2 is pretty low although the hyperparameters is good.