import pandas as pd

def read_csv():
    df_cars = pd.read_csv('../DATA/cars.csv')
    print(df_cars)

    print()

    df_cars_indexed = pd.read_csv('../DATA/cars.csv', index_col = 0)
    print(df_cars_indexed)


if __name__ == '__main__':
    read_csv()
