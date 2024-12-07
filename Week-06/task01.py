import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    music_to_json = pd.read_json("../DATA/music_dirty.txt")
    music_df = pd.DataFrame(music_to_json)

    print(f'Shape before one-hot-encoding: {np.shape(music_df)}')

    music_dummies = pd.get_dummies(music_df['genre'], drop_first=True, dtype=int)
    music_dummies = pd.concat([music_df, music_dummies], axis=1)

    f, ax = plt.subplots(figsize = (10,5), sharey=True)
    music_dummies.boxplot(column='popularity', by='genre', ax=ax, patch_artist=True, grid=False)
    plt.tight_layout()
    plt.show()

    music_dummies = music_dummies.drop(columns=['genre']) 
    print(f'Shape after one-hot-encoding: {np.shape(music_dummies)}')

if __name__ == '__main__':
    main()