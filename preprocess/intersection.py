from sys import base_exec_prefix
import pandas as pd 
import torch
from torch import nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import torch.nn.functional as F
import os

'''
    整合所有样本, 对expr取交集
'''
path = './data/2'

expr = pd.read_csv('data/2/Discovery_cohort2/Riaz/Riaz_exp.csv', sep = ',', index_col=0, encoding='ISO-8859-1')
print(expr)
clin = pd.read_csv('data/2/Discovery_cohort2/Riaz/Riaz_clin.csv', sep=',', index_col=0, encoding='ISO-8859-1')
# print(clin)

# expr_2 = pd.read_csv('data/2/Discovery_cohort2/Liu/Liu_exp.csv', sep = ',', index_col=0, encoding='ISO-8859-1')
# print(expr_2)
# clin_2 = pd.read_csv('data/2/Discovery_cohort2/Liu/Liu_clin.csv', sep=',', index_col=0, encoding='ISO-8859-1')
# # print(clin_2)

# expr_merge = pd.concat([expr, expr_2], axis=1, join='inner')
# print(expr_merge)
# clin_merge = pd.concat([clin, clin_2], axis=0)
# print(clin_merge)
# print(clin_merge['response'])

# gene_set = set(expr_table.iloc[:, 0])

for folder in os.listdir(path):
    if folder != 'Validation_cohort':
        for cohort in os.listdir(os.path.join(path, folder)):
            if cohort not in ['Hwang', 'Jerby_Arnon', 'Riaz']:
                # print(expr.shape[0], expr.shape[1])
                data_expr = pd.read_csv(f'./data/2/{folder}/{cohort}/{cohort}_exp.csv', sep=',', index_col=0, encoding='ISO-8859-1')
                data_clin = pd.read_csv(f'./data/2/{folder}/{cohort}/{cohort}_clin.csv', sep=',', index_col=0, encoding='ISO-8859-1')
                expr = pd.concat([expr, data_expr], axis=1, join='inner')
                # print(expr)
                clin = pd.concat([clin, data_clin], axis=0, join='inner')
                # print(data_expr.head())
                # print(expr.shape[0], expr.shape[1])
                # print(clin.shape[0], clin.shape[1])
                # print(len(gene_set))

clin = clin.drop_duplicates()
expr.to_csv("data/3/merged/merged/merged_exp.csv", sep=',')
clin.to_csv("data/3/merged/merged/merged_clin.csv", index=0, sep=',')
print(expr.shape)
print(clin.shape)

# TODO: 收集所有的 expr 和 clin 信息合为一个文件夹
# for folder in os.listdir(path):
#     for cohort in os.listdir(os.path.join(path, folder)):
#         if cohort not in ['Hwang', 'Jerby_Arnon']: