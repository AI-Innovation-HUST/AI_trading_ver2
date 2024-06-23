from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Binarizer, QuantileTransformer, PowerTransformer, MaxAbsScaler, Normalizer
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ScalerData(BaseEstimator, TransformerMixin):
    def __init__(self, method='minmax'):
        self.method = method
        self.scaler = None

    def fit(self, X, y=None):
        if self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'zscore':
            self.scaler = StandardScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        elif self.method == 'log':
            self.scaler = None  # Log transformation doesn't use a scaler
        elif self.method == 'binarizer':
            self.scaler = Binarizer()
        elif self.method == 'quantile':
            self.scaler = QuantileTransformer()
        elif self.method == 'power':
            self.scaler = PowerTransformer()
        elif self.method == 'maxabs':
            self.scaler = MaxAbsScaler()
        elif self.method == 'normalize':
            self.scaler = Normalizer()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        if self.scaler is not None:
            self.scaler.fit(X)

        return self

    def transform(self, X):
        if self.method in ['minmax', 'zscore', 'robust', 'binarizer', 'quantile', 'power', 'maxabs', 'normalize']:
            X_scaled = self.scaler.transform(X)
        elif self.method == 'log':
            X_scaled = X.apply(lambda x: np.log1p(x))
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def calculate(self, df):
        # Select numerical column
        numerical_data = df.select_dtypes(include=[np.number]).to_numpy()
        mean = np.mean(numerical_data)
        std = np.std(numerical_data)
        # Print test results
        print("mean:", mean)
        print("std:", std)
        return mean, std




# # Test case
# if __name__ == "__main__":

#     data = pd.read_csv('C:\\Users\\7400\\OneDrive\\Code\\AI_tools_trading\\AI_tools\\Dataset\\test.csv')
#     df = pd.DataFrame(data)

#     scaler = ScalerData(method='minmax')
#     df_scaled = scaler.fit_transform(df)
#     print("Min-Max Scaled Data:\n", df_scaled)

#     scaler = ScalerData(method='zscore')
#     df_scaled = scaler.fit_transform(df)
#     print("Z-score Scaled Data:\n", df_scaled)

#     scaler = ScalerData(method='log')
#     df_scaled = scaler.fit_transform(df)
#     print("Log Transformed Data:\n", df_scaled)

#     scaler = ScalerData(method='robust')
#     df_scaled = scaler.fit_transform(df)
#     print("Robust Scaled Data:\n", df_scaled)
 
#     scaler = ScalerData(method='binarizer')
#     df_scaled = scaler.fit_transform(df)
#     print("Binarized Data:\n", df_scaled)

#     scaler = ScalerData(method='quantile')
#     df_scaled = scaler.fit_transform(df)
#     print("Quantile Transformed Data:\n", df_scaled)
  
#     scaler = ScalerData(method='power')
#     df_scaled = scaler.fit_transform(df)
#     print("Power Transformed Data:\n", df_scaled)
   
#     scaler = ScalerData(method='maxabs')
#     df_scaled = scaler.fit_transform(df)
#     print("MaxAbs Scaled Data:\n", df_scaled)
    
#     scaler = ScalerData(method='normalize')
#     df_scaled = scaler.fit_transform(df)
#     print("Normalized Data:\n", df_scaled)
