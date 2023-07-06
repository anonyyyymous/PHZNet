import random
import numpy as np
from sklearn.neighbors import NearestNeighbors    # k近邻算法

'''
    https://blog.csdn.net/qq_42374697/article/details/118412555#:~:text=SMOTE%20-Pytorch%EF%BC%9A%20SMOTE%20%E7%9A%84Pytorch%20%E5%AE%9E%E7%8E%B0,02-09%20SMOTE%20%EF%BC%9A%E7%BB%BC%E5%90%88%E5%B0%91%E6%95%B0%E6%97%8F%E8%A3%94%E8%BF%87%E9%87%87%E6%A0%B7%E6%8A%80%E6%9C%AF%20%E5%85%B3%E4%BA%8E%20%E5%A6%82%E6%9E%9C%E5%88%86%E7%B1%BB%E6%A0%87%E7%AD%BE%E7%9A%84%E5%88%86%E5%B8%83%E4%B8%8D%E5%9D%87%E7%AD%89%EF%BC%8C%E5%88%99%E6%95%B0%E6%8D%AE%E9%9B%86%E5%B0%86%E5%A4%84%E4%BA%8E%E4%B8%8D%E5%B9%B3%E8%A1%A1%E7%8A%B6%E6%80%81%EF%BC%8C%E5%9B%A0%E6%AD%A4%EF%BC%8C%E5%9C%A8%E8%AF%B8%E5%A6%82%E6%AC%BA%E8%AF%88%E6%A3%80%E6%B5%8B%E4%B9%8B%E7%B1%BB%E7%9A%84%E5%A4%A7%E9%87%8F%E7%8E%B0%E5%AE%9E%E4%B8%96%E7%95%8C%E4%B8%AD%EF%BC%8C%E5%B8%B8%E8%A7%81%E7%9A%84%E9%97%AE%E9%A2%98%E6%98%AF100%E5%88%B01%E7%9A%84%E4%B8%8D%E5%B9%B3%E8%A1%A1%E3%80%82
'''

class Smote:
    def __init__(self,samples,N,k):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)    # 1.对每个少数类样本均求其在所有少数类样本中的k近邻
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic
	# 2.为每个少数类样本选择k个最近邻中的N个；3.并生成N个合成样本
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
