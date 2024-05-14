import os
import pandas as pd
from glob import glob
import re
import torch
from torch.utils.data import Dataset#, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')



class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.data_x, self.data_y, self.data_stamp, self.data_block = [], [], [], []
        total_len = 0

        for filename in glob(f"{self.root_path}/*.csv"):

            self.scaler = StandardScaler()
            df_raw = pd.read_csv(filename)

            '''
            df_raw.columns: ['date', ...(other features), target feature]
            '''
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
            num_test = min(int(len(df_raw) * 0.2), 1000) # 数据过大时，限制绝对数量
            num_vali = min(int(len(df_raw) * 0.1), 500)
            num_train = len(df_raw) - num_vali - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features == 'M' or self.features == 'MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features == 'S':
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_x.append(data[border1:border2])
            #self.data_y.append(data[border1:border2])
            self.data_stamp.append(data_stamp)
        
            total_len = total_len + len(self.data_x[-1]) - self.seq_len - self.pred_len
            self.data_block.append((len(self.data_block), total_len))


    def get_block_index(self, blocks, index):
        if len(blocks)==1:
            return blocks[0][0]
        mid = len(blocks) // 2 - 1
        if index > blocks[mid][1]:
            return self.get_block_index(blocks[mid+1:], index)
        else:
            return self.get_block_index(blocks[:mid+1], index)


    def __getitem__(self, index0):
        # 定位数据块
        block = self.get_block_index(self.data_block, index0)
        if block > 0:
            index = index0 - self.data_block[block-1][1]
        else:
            index = index0

        assert index0<=self.data_block[-1][1], "index out of range"

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[block][s_begin:s_end]
        seq_y = self.data_x[block][r_begin:r_end]
        seq_x_mark = self.data_stamp[block][s_begin:s_end]
        seq_y_mark = self.data_stamp[block][r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return self.data_block[-1][1]


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
