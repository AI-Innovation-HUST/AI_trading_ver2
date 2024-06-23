import random
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader,Dataset
import ast
from numpy import load
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import torch
import random
import pandas as pd

class CoinDataset(Dataset):
    def __init__(self, data_frame):
        self.data_frame = data_frame
        

    def __len__(self):
        # Calculate the number of samples considering potential padding
        data_len = len(self.data_frame)
        return data_len - (data_len % 16)  # Subtract remainder to ensure divisibility by 16

    def __getitem__(self, idx):
        sample = []
        target = []
        '''
        Quy đổi theo công thức Z= X-u/xich_ma 
        trong đó u là mean 
        xich_ma là phân bố chuẩn
        '''
        sample = []
        target = []
        
        # Normalize each column value
        for column in self.data_frame.columns:
            if column not in ['ts','h','l']:  # Skip 'ts' column if it's not needed
                sample.append(self.data_frame[column].iloc[idx])
        
        target.append([self.data_frame['h'].iloc[idx+1],self.data_frame['l'].iloc[idx+1]])
        # list = ['123']
        # sample.append(abs((self.data_frame['c'][idx]-self.mean))/self.std)
        # target.append([(abs((self.data_frame['h'][idx+1]-self.mean))/self.std),(abs((self.data_frame['l'][idx+1]-self.mean))/self.std)])
        
        # change value to tensor

        sample = torch.tensor(sample)  # Add leading dimension for [1, 16]
        target = torch.tensor(target)  # Add leading dimension for [1, 16]


        return sample, target

# # Ví dụ về cách sử dụng CoinDataset và DataLoader
# if __name__ == "__main__":
#     import pandas as pd 
#     import numpy as np
#     df = pd.read_csv('Dataset/test.csv')
#     import pandas as pd

#     # Load CSV data into a DataFrame

#     # Select numerical column
#     numerical_data = df['o'].to_numpy()
#     mean = np.mean(numerical_data)
#     std = np.std(numerical_data)

 

#     # Print test results
#     print("mean:",np.mean(numerical_data))
#     print("std:", np.std(numerical_data))
#     coin_dataset = CoinDataset(df,mean=mean,std=std)
#     dataloader = DataLoader(coin_dataset, batch_size=16, shuffle=True, num_workers=0)
    
#     for data in dataloader:
#         d,t = data
#         print(d)
#         break
