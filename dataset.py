import pandas as pd 

import torch
from torch.utils.data import Dataset


'''
    直接读取 csv 文件的原始版本
'''
class predictio_dataset(Dataset):
    """
    Loads the PredictIO dataset
    """
    def __init__(self, folder, cohort, split='train'):
        # 读取诊断信息
        self.data_clin = pd.read_csv(f'./data/{folder}/{cohort}/{cohort}_clin.csv', sep=',', encoding='ISO-8859-1')
        # self.data_clin.set_index('patient')
        self.data_clin.index = self.data_clin['patient']
        # 消除所有 response 中的 NaN
        self.data_clin = self.data_clin.drop(self.data_clin[(self.data_clin.response!=self.data_clin.response)].index)

        self.data_expr = pd.read_csv(f'./data/{folder}/{cohort}/{cohort}_exp.csv', sep=',', index_col=0, encoding='ISO-8859-1')
        # print(self.data_expr[self.data_expr.isnull().values==True])
        self.data_expr = self.data_expr.dropna(axis=1, how='any')
        print(self.data_expr.shape)
        # print(self.data_expr[self.data_expr.isnull().values==True])
        # print(type(self.data_expr.isnull().any())) 
        # for idx, i in enumerate(self.data_expr.isnull().any()):
        #     if i:
        #         print('true')
        #         print(self.data_expr)

        self.name_list = list(set(self.data_expr.columns[1:]).intersection(self.data_clin['patient']))
        # print(self.data_expr.columns)
        # print(self.data_clin['patient'])
        # print(self.name_list)
        print(f'In {cohort}, there are {len(self.name_list)} samples and {self.data_expr.shape[0]} params')
        self.n_features = self.data_expr.shape[0]
        # print(self.data_expr.head())
        # print(self.data_clin.head())
        # print(self.data_clin.loc['PD_015T'])
        # print(self.data_expr['PD_015T'])
        # torch.from_numpy(data_expr.loc['RCC25-677'].values).type(torch.float32)
        # self.data_expr = self.data_expr.transpose()
        # print(self.data_expr.head())
        # print(self.data_clin.head())

                    
    def __len__(self):
        # return len(self.data_clin)
        return len(self.name_list)
        
    def __getitem__(self, idx):
        # patient = self.data_clin['patient'][idx]
        # return [torch.from_numpy(self.data_expr.loc[patient].values).type(torch.float32).unsqueeze(0), \
        #         int(self.data_clin['response'][idx]=='R')]
        # print(self.data_expr[self.name_list[idx]].dtypes)
        if isinstance(self.data_clin.loc[self.name_list[idx]]['response'], pd.core.series.Series):
            return [torch.from_numpy(self.data_expr[self.name_list[idx]].values).type(torch.float32), \
                    int(self.data_clin.loc[self.name_list[idx]].iloc[0]['response']=='R')]
        else:
            return [torch.from_numpy(self.data_expr[self.name_list[idx]].values).type(torch.float32), \
                    int(self.data_clin.loc[self.name_list[idx]]['response']=='R')]
    