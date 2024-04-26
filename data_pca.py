import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.manifold import SpectralEmbedding

#读取文件
pdata = pd.read_csv("train_data_standard_a.csv", header=0, index_col=None)
X = pdata.iloc[:, 0:14].values
y = pdata.iloc[:, 14].values
print(X.shape)
print(y.shape)

# #进行PCA变换
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

## 拉普拉斯特征映射
# spe = SpectralEmbedding(n_components=2)
# X_r = spe.fit_transform(X)

#画散点图
plt.figure()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
lw = 2 

for color, i in zip(colors, [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13]):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw)
plt.title("PCA of company dataset")
plt.show()
