import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def data_standardization(in_file,out_file,columns_select):
    df = pd.read_csv(in_file)
    select_df = df[columns_select]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(select_df)
    scaled_df = pd.DataFrame(scaled_data,columns=select_df.columns)
    scaled_df.to_csv(out_file,index=False)
    
in_csv = 'train_data_features.csv'
out_csv = 'train_data_standard_a.csv'
columns_select = ['JXFP','XXFP','JXJE','XXJE','FFP','FPZS','ZFFP','JXJS','XXJS','RM','LT','WD','YL','ML']
data_standardization(in_csv,out_csv,columns_select)
