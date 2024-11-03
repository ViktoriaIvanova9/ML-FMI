import pandas as pd

def read_cars_advances():
    df_cars_indexed = pd.read_csv('../DATA/cars_advanced.csv', index_col = 0)

    print(df_cars_indexed[ df_cars_indexed['drives_right'] == True])
    print()
    print(df_cars_indexed[df_cars_indexed['cars_per_cap'] > 500]['country'])
    print() 
    print(df_cars_indexed[(df_cars_indexed['cars_per_cap'] >= 10) & (df_cars_indexed['cars_per_cap'] <= 80)]['country'])
    print()
    print(df_cars_indexed[df_cars_indexed['cars_per_cap'].between(10, 80)]['country'])

    
if __name__ == '__main__':
    read_cars_advances()