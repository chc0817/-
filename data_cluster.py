import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

pdata = pd.read_csv("train_data_standard.csv", header=0, index_col=None)
X = pdata.iloc[:, 9:14].values
y = pdata.iloc[:, 14].values
print(X.shape)
print(y.shape)

# pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
spe = SpectralEmbedding(n_components=2)
X_r = spe.fit_transform(X)

range_n_clusters = [2, 3, 4, 5, 6]
silhouette_avg_list = []
for n_clusters in range_n_clusters:
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X_r)

        silhouette_avg = silhouette_score(X_r, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)


plt.figure()
plt.plot(range_n_clusters, silhouette_avg_list)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_score")
plt.title("Spectral of cluster numbers")
plt.show()        