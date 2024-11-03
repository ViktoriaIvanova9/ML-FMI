import pandas as pd

def print_rows():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)
    
    df_cars_indexed['COUNTRY'] = df_cars_indexed['country']

    print(df_cars_indexed)

if __name__ == '__main__':
    print_rows()