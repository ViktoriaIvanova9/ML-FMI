import pandas as pd

def read_cars():
    df_cars = pd.read_csv('../DATA/cars.csv', index_col = 0)
    print(df_cars['country'])
    print()
    print(df_cars[['country']])
    print()
    print(df_cars[['country', 'drives_right']])
    print()
    print(df_cars[0:3])
    print()
    print(df_cars[3:6])

def read_cars_advanced():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)

    print(df_cars_indexed.loc['JPN'])
    print(type(df_cars_indexed.loc['JPN']))
    print()
    print(df_cars_indexed.loc[['AUS', 'EG']])
    print(type(df_cars_indexed.loc[['AUS', 'EG']]))
    print()
    print(df_cars_indexed.loc[['MOR'], ['drives_right']])
    print()
    print(df_cars_indexed.loc[['RU', 'MOR'], ['country', 'drives_right']])


if __name__ == '__main__':
    read_cars()
    read_cars_advanced()