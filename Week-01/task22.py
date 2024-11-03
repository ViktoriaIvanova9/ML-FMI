import pandas as pd

def create_dataframe():
    names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
    dr =  [True, False, False, False, True, True, True]
    cpc = [809, 731, 588, 18, 200, 70, 45]

    data = {'country' : names,
            'drives_right' : dr,
            'cars_per_cap' : cpc}

    df_vehicle = pd.DataFrame(data)
    df_vehicle.index = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']
    print(df_vehicle)

if __name__ == '__main__':
    create_dataframe()