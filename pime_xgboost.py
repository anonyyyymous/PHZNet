import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
from xgboost import XGBClassifier, plot_importance
from torch import nn
import matplotlib.pyplot as plt
from model.phznet import PHZNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import random


def pipeline(seed):
    print("seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载并初始化你的PHZNet模型
    phznet = PHZNet()
    phznet.load_state_dict(torch.load('./model/models/model_PHZNet_0.001_32.pth'))
    phznet = phznet.to(device)
    phznet.eval()  # 将模型设置为评估模式

    # 创建 PyTorch 数据集的类
    class GeneDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
        

    data_csv = pd.read_csv('./data/3/augmented/augmented/augmented_exp.csv', index_col=0, sep=',', encoding='ISO-8859-1').transpose()  # 你的数据文件路径
    label_csv = pd.read_csv('./data/3/augmented/augmented/augmented_clin.csv', sep=',', encoding='ISO-8859-1')  # 你的标签文件路径

    df = pd.merge(data_csv, label_csv[['patient', 'response']], left_index=True, right_on='patient')
    # df.set_index('patient', inplace=True)

    df = df.dropna()  # 删除 NaN 的行
    # print(df)
    # print(df.shape)

    le = LabelEncoder()
    df['response'] = le.fit_transform(df['response'])  # 对 'response' 列进行编码

    X = df.drop(['patient', 'response'], axis=1).values
    y = df['response'].values

    # 加载数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 分割数据为训练集和测试集

    # 转换为 Tensor
    X_train, y_train = torch.from_numpy(X_train.astype('float32')).to(device), torch.from_numpy(y_train.astype('long')).to(device)
    X_test, y_test = torch.from_numpy(X_test.astype('float32')).to(device), torch.from_numpy(y_test.astype('long')).to(device)

    # 初始化 PHZNet
    phznet = PHZNet().to(device)

    # 用 PHZNet 生成新的标签
    with torch.no_grad():
        y_train = torch.max(phznet(X_train), 1)[1]
        y_test = torch.max(phznet(X_test), 1)[1]

    # 将 Tensor 转换回 numpy，以便可以用于 DecisionTreeClassifier
    X_train, y_train = X_train.cpu().numpy(), y_train.cpu().numpy()
    X_test, y_test = X_test.cpu().numpy(), y_test.cpu().numpy()

    # 创建 PyTorch 数据集
    train_dataset = GeneDataset(X_train, y_train)
    test_dataset = GeneDataset(X_test, y_test)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    model = XGBClassifier(use_label_encoder=False)
    # 对于每个 batch
    for batch_idx, (data, target) in enumerate(train_loader):
        # 在这里，你可以使用你的模型进行训练，例如：
        model.fit(data, target)

    ####################################################################################
    # 验证模型效果

    from sklearn.metrics import accuracy_score, classification_report

    # 创建 DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    truths = []

    # 在测试集上测试模型
    for batch_idx, (data, target) in enumerate(test_loader):
        pred = model.predict(data.cpu().numpy())  # 计算模型预测
        predictions.extend(pred)
        truths.extend(target.cpu().numpy())

    print("Accuracy:", accuracy_score(truths, predictions))  # 计算准确率
    print(classification_report(truths, predictions))  # 输出分类报告

    # plot_importance(model)
    # plt.savefig('feature_importance.png')  # 保存图像到文件

    # 获取特征的重要性
    importance_scores = model.feature_importances_

    # 获取特征的名字
    feature_names = df.drop(['response'], axis=1).columns

    # 创建一个字典，其中键是特征名字，值是特征的重要性
    feature_importances = dict(zip(feature_names, importance_scores))

    # 按照特征的重要性从高到低排序
    sorted_feature_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    # 只打印前20个最重要的特征
    for feature, importance in sorted_feature_importances[:200]:
        if importance < 0.00001:
            break
        print(f"Feature: {feature}, Importance: {importance}")

    print("\n\n\n")
    # 只打印前20个最重要的特征
    for feature, importance in sorted_feature_importances[:200]:
        if importance < 0.00001:
            break
        print(f"{feature}")

for i in range (200, 500):
    try:
        pipeline(i)
    except:
        continue
