import csv
 
out = open("/data/home/jzheng/hzpan/PredictIO/data/3/before_merge/before_merge/before_merge.csv", 'w', newline='')
csv_writer = csv.writer(out,dialect='excel')  
 
f = open("/data/home/jzheng/hzpan/PredictIO/data/3/before_merge/before_merge/before_merge.txt", "r")
for line in f.readlines()[1:]:
     list = line.split()          #将字符串转为列表，从而可以按单元格写入csv
     csv_writer.writerow(list)

exprfile_list = ['/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort3/Mariathasan/Mariathasan_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort3/Liu/Liu_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort1/Hugo/Hugo_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort1/Fumet2/Fumet2_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort2/Jung/Jung_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort2/Snyder/Snyder_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort2/Riaz/Riaz_exp.csv',
    '/data/home/jzheng/hzpan/PredictIO/data/2/Discovery_cohort1/Braun/Braun_exp.csv']

name_list = ['']

for exprfile in exprfile_list:
    expr = pd.read_csv(exprfile, sep=',', index_col=0, encoding='ISO-8859-1')
    name_list += list(expr.columns.values)
    # print(expr.index)
    # print(len(expr.columns))

# print(name_list)
# print(len(name_list))

name_list = pd.DataFrame(name_list).T

expr = pd.read_csv("/data/home/jzheng/hzpan/PredictIO/data/3/before_merge/before_merge/before_merge.csv", sep=',', encoding='ISO-8859-1', header=None)
print(name_list.columns, name_list.shape)
print(expr.columns)
expr = pd.concat([name_list, expr], axis=0, ignore_index=True)
# print(expr)
expr.to_csv("/data/home/jzheng/hzpan/PredictIO/data/3/before_merge/before_merge/before_merge_exp.csv", sep=',', header=None, index=None)