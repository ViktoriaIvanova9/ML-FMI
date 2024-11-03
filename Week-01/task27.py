import pandas as pd

def print_rows():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)
    
    for val, row in df_cars_indexed.iterrows():
        print(f'{val}: {row.iloc[0]}')

if __name__ == '__main__':
    print_rows()