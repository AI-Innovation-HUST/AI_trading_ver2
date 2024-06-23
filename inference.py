from model import *
import torch
import pandas as pd
import pickle
import sys
import numpy as np
from scaler import *

sys.setrecursionlimit(10000) # 10000 is an example, try with different values

import ta

class Inference():
    
    def __init__(self,model_path='evalModel_best.pth',scaler='result_transformer/scaler.pkl',device = 'cpu'):
        self.model = torch.load(model_path,map_location="cpu")
        self.model.to(device)
        self.scaler =  pickle.load(open(scaler, 'rb'))
        
        
    def inference(self,df):
        
        df = ta.add_all_ta_features(df, "o", "h", "l", "c", "vol", fillna=True)
        scaler = MinMaxScaler() # test method
        df['prediction_h'] = np.ones((df.shape[0],1))
        df['prediction_l'] = np.ones((df.shape[0],1))

        
                                        
        df_norm = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df_norm.dropna(inplace=True)
        HIGH = []
        LOW = []
        
        # Normalize each column value
        for idx in range(df.shape[0]):
            sample = []
            for column in df.columns:
                if column not in ['ts','h','l','prediction_h','prediction_l']:  # Skip 'ts' column if it's not needed
                    sample.append(df_norm[column].iloc[idx])
            data = torch.tensor(sample)
            output = self.model(data.unsqueeze(0))
            output = output.view(-1,2)
            output = output.detach().numpy()[0]
            print(output)

            HIGH.append(output[1])
            LOW.append(output[0])
        df_norm["prediction_h"] = HIGH
        df_norm ["prediction_l"] = LOW
        df_reverted = df_norm.copy()
        df_reverted.to_csv("results.csv")

        columns_to_reverse = ['o', 'l']

        df_reverted[columns_to_reverse] = scaler.inverse_transform(df_norm)[:, 2:3]

            

if __name__ =="__main__":
    model = Inference()
    df = pd.read_csv("Dataset/data_1714496400000_1717174800000.csv")
    out = model.inference(df)
    
