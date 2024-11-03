import pandas as pd

def print_rows():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)

    for label, row in df_cars_indexed.iterrows():
        print(f'Label is: "{label}"')
        print('Row contents: ')
        print(row)
        print()    

if __name__ == '__main__':
    print_rows()