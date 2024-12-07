import json
import numpy as np
import pandas as pd

def main():
    with open("../DATA/music_dirty_missing_vals.txt") as music_file:
        music_to_json = json.load(music_file)

    music_df = pd.DataFrame(music_to_json)

    print(f'Shape of input dataframe: {np.shape(music_df)}')
    print("Percentage of missing values:")
    print(music_df.isna().mean().sort_values(ascending=False))

    missing = music_df.isna().mean().sort_values(ascending=False)
    drop_from_columns = missing[missing < 0.05].index.to_list()
    music_df = music_df.dropna(subset=drop_from_columns)
    print(f'Columns/Variables with missing values less than 5% of the dataset: {drop_from_columns}')

    music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
    print('First five entries in `genre` column:')
    print(music_df['genre'][:5])

    print(f'Shape of preprocessed dataframe: {np.shape(music_df)}')

if __name__ == '__main__':
    main()