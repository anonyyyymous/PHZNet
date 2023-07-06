import pandas as pd 
import os

data_path = './data/2'

for f1 in ['Discovery_cohort1', 'Discovery_cohort2', 'Discovery_cohort3', 'Validation_cohort']:
    print(os.listdir(os.path.join(data_path, f1)))
    for f2 in os.listdir(os.path.join(data_path, f1)):
        if f2 != '.DS_Store':
            file = f"{data_path}/{f1}/{f2}/{f2}_exp"
            print(file)
            expr = pd.read_csv(f"{file}.csv", sep=',', encoding='ISO-8859-1')
            expr = expr.dropna(axis=1, how='any')
            with open(f"{file}.txt", 'w+', encoding='utf-8') as f:
                # f.write("\t".join(expr.columns.values.tolist()[1:])+'\n')
                for line in expr.values:
                    for i in range(len(line)):
                        f.write((str(line[i])))
                        if i != len(line)-1:
                            f.write('\t')
                        else:
                            f.write('\n')

                    # print("/t".join(line))
                    # print((str(line[0])+'\t'+str(line[1])+'\n'))
                    # f.write((str(line[0])+'\t'+str(line[1])+'\n'))


# file = f"data/{folder}/{cohort}/{cohort}_exp"
# expr = pd.read_csv(f"{file}.csv", sep=',', encoding='ISO-8859-1')
# print(expr.shape)
# # print(expr.columns)
# expr = expr.drop_duplicates(['Unnamed: 0'])
# print(expr['Unnamed: 0'])
# print(expr.shape)
# print(expr.loc['1-Mar'])