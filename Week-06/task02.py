import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error

def main():
    with open("../DATA/music_dirty.txt") as music_file:
        music_to_json = json.load(music_file)

    music_df = pd.DataFrame(music_to_json)
    music_dummies = pd.get_dummies(music_df['genre'], drop_first=True, dtype=int)
    music_dummies = pd.concat([music_df, music_dummies], axis=1)
    music_dummies = music_dummies.drop(columns=['genre'])

    X = music_dummies.drop('popularity', axis=1)
    y = music_dummies['popularity'].values

    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    ridge_reg = Ridge(alpha=0.2)
    y_scores = cross_val_score(ridge_reg, X, y, scoring='neg_root_mean_squared_error', cv=kf)

    print(f'Average RMSE: {np.mean(np.abs(y_scores))}')
    print(f'Standard Deviation of the target array: {np.std(y)}')

if __name__ == '__main__':
    main()

    # Given the value of the average RMSE and the standard deviation of the target column, does the model perform well? - 
    # yes, it is reducing from 14 to 8 which is a good improvement