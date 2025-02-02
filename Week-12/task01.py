import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class WaterDataset(Dataset):
    def __init__(self):
        super(WaterDataset, self).__init__()
        self.water_ds = np.array(pd.read_csv('../DATA/water_train.csv'))

    def __len__(self):
        return len(self.water_ds)

    def __getitem__(self, position):
        return (self.water_ds[position][:-1], self.water_ds[position][-1])
        
def main():
    water_potability = WaterDataset() 
    print(f'Number of instances: {len(water_potability)}')
    print(f'Fifth item: {water_potability[4]}')

    train_dataloader = DataLoader(water_potability, batch_size=2, shuffle=True)
    print(next(iter(train_dataloader))) # isn't [0., 0.] the first batch?

if __name__ == '__main__':
    main()