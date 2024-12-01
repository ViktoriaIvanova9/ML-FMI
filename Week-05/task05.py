import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV

def main():
    diabetes_df = pd.read_csv('../DATA/diabetes_clean.csv')

    X = diabetes_df.drop('diabetes', axis=1)
    y = diabetes_df['diabetes'].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    param_grid = {
        'penalty' : np.array(['l1', 'l2']),
        'tol' : np.linspace(0.0001, 1.0, 50),
        'C' : np.linspace(0.1, 1.0, 50),
        'class_weight' : ['balanced', {0:0.8, 1:0.2}]
    }

    kf = KFold(n_splits=6, shuffle=True, random_state=42)

    log_reg = RandomizedSearchCV(LogisticRegression(solver='saga'), param_grid, cv=kf, random_state=42).fit(X_train, y_train)

    print(f'Tuned Logistic Regression Parameters: {log_reg.best_params_}')
    print(f'Tuned Logistic Regression Best Accuracy Score: {log_reg.best_score_}')


if __name__ == '__main__':
    main()