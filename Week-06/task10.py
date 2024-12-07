import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    with open("../DATA/music_dirty_missing_vals.txt") as music_file:
        music_to_json = json.load(music_file)

    music_df = pd.DataFrame(music_to_json)
    # music_dummies = pd.get_dummies(music_df['genre'], drop_first=True, dtype=int)
    # music_dummies = pd.concat([music_df, music_dummies], axis=1)
    # music_dummies = music_dummies.drop(columns=['genre'])

    X = music_df.drop('genre', axis=1)
    music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
    y = music_df['genre'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    param_grid = {
        'logreg__solver' : ['newton-cg', 'saga', 'lbfgs'],
        'logreg__C' : np.linspace(0.001, 1.0, 10)
    }

    steps = [('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler()),
             ('logreg', LogisticRegression())]
    pipeline = Pipeline(steps)
    logreg = GridSearchCV(pipeline, param_grid).fit(X_train, y_train)

    logreg_scaling_score = logreg.score(X_test, y_test)

    print(f'Tuned Logistic Regression Parameters: {logreg.best_params_}')
    print(f'Accuracy: {logreg_scaling_score}')


if __name__ == '__main__':
    main()