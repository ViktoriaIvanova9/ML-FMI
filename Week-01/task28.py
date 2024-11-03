import pandas as pd

def print_rows():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)
    
    for val, _ in df_cars_indexed.iterrows():
        df_cars_indexed.loc[val, 'COUNTRY'] = df_cars_indexed.loc[val, 'country'].upper()
        
    print(df_cars_indexed)

if __name__ == '__main__':
    print_rows()