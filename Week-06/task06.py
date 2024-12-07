import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

def main():
    music_df = pd.read_csv('../DATA/music_clean.csv', index_col=0)

    X = music_df.drop('genre', axis=1)
    y = music_df['genre'].values # here by the description I was left with the impression that I should use np.where to check if it is Rock
                                 # to set it to 1 and if no 0 and probably my mistake that I didn't have looked at the data before using it
                                 # but spent a lot of time trying to see why it is failing and it was because 'Rock' wasn't part of the data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
    parameters = {'logreg__C': np.linspace(0.001, 1.0, 20)}

    steps = [('logreg', LogisticRegression(random_state=21))]
    pipeline = Pipeline(steps)
    logreg = GridSearchCV(pipeline, parameters).fit(X_train, y_train)
    logreg_score = logreg.score(X_test, y_test)

    print(f'Without scaling: {logreg_score}')
    print(f'Without scaling: {logreg.best_params_}') # without scaling the best parameter is different


    steps_scaling = [('scaler', StandardScaler()),
                    ('logreg', LogisticRegression(random_state=21))]
    pipeline_scaling = Pipeline(steps_scaling)

    logreg_scale = GridSearchCV(pipeline_scaling, parameters).fit(X_train, y_train)
    logreg_scaling_score = logreg_scale.score(X_test, y_test)

    print(f'With scaling: {logreg_scaling_score}')
    print(f'With scaling: {logreg_scale.best_params_}')

if __name__ == '__main__':
    main()

    # do the models perform well? - with scaling kind of yes