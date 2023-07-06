from preprocess.smote import Smote
import pandas as pd
import numpy as np

# 生成expr
# 读取expr
expr = pd.read_csv("data/3/merged/merged/merged_exp.csv", sep=',', index_col=0, encoding='ISO-8859-1')
expr = expr.dropna(axis=1, how='any')
print(expr.shape)   # (16093, 936)，其中NR 513例，R 251例

# 读取clin并筛选其中['response']=='R'的
clin = pd.read_csv("data/3/merged/merged/merged_clin.csv", sep=',', encoding='ISO-8859-1')
rp = clin[clin['response']=='R']['patient']
not_in_rp = pd.DataFrame(['P1494', 'P7623', 'P6126', 'P6336', 'P9699', 'P346', 'P2056', 'P167', 'P9705', 'P1509', 'P57', 'Pat91', 'Pat39', 'PD_018T', 'Pat79', 'Pat88', 'P18', 'Pat38', 'Pat04', 'PD_023T', 'PD_019T', 'P19', 'RCC_106', 'P35', 'PD_025T', 'P6', 'Pat126', 'Pat90', 'P8', 'P54', 'PD_007T', 'PD_013T', 'P51', 'P59', 'P21', 'P56', 'Pat47', 'PD_010T'])
diff = pd.concat([rp,not_in_rp,not_in_rp]).drop_duplicates(keep=False).values.squeeze()

# 使用smote算法生成新的数据
newexpr = Smote(expr[diff].values.transpose(),1,5).over_sampling().transpose()
# print(newexpr)
print(newexpr.shape)    # (16093, 237)，因为实际能找到expr的R只有237例，所以只能生成237例

# 转成dataframe并append到原expr中
newexpr_dataframe = pd.DataFrame(newexpr, index=expr.index)
expr = pd.concat([expr, newexpr_dataframe], axis=1, join='inner')
print(expr.shape)   # (16093, 1173)
# print(expr)

# expr.to_csv("data/3/augmented/augmented/augmented_exp.csv", index=True, sep=',')


# 生成对应clin
namelist = [str(x) for x in range(1, newexpr.shape[1]+1)]
reponselist = ['R']*newexpr.shape[1]
# print(type(namelist), type(reponselist))
data = {'patient':namelist,'response':reponselist} # 两组列元素，并且个数需要相同
df = pd.DataFrame(data) # 这里默认的 index 就是 range(n)，n 是列表的长度

clin = pd.read_csv("data/3/merged/merged/merged_clin.csv", sep=',', encoding='ISO-8859-1')
clin = clin.append(df, ignore_index = True)
print(clin.shape)

clin.to_csv("data/3/augmented/augmented/augmented_clin.csv", index=0, sep=',')