import math
import pandas as pd

folder = 'augmented'
# folder = 'merged'

expr = pd.read_csv(f"../data/3/{folder}/{folder}/{folder}_exp.csv", sep=',', index_col=0, encoding='ISO-8859-1')
# expr = expr.applymap(lambda x: math.log(x+1))
# expr.to_csv("data/3/merged_log/merged_log/merged_log_exp.csv", sep=',')

clin = pd.read_csv(f"../data/3/{folder}/{folder}/{folder}_clin.csv", sep=',', encoding='ISO-8859-1')


print(len(clin[clin['response']=='NR']))
print(len(clin[clin['response']=='R']))
print(clin[clin['response']=='R'].head())
print(clin[clin['response']=='R']['patient'])
# print(clin[clin['response']=='R'])