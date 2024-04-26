import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

pdata = pd.read_csv("train_data_standard.csv", header=0, index_col=None)
X = pdata.iloc[:, 9:14].values
y = pdata.iloc[:, 14].values
print(X.shape)
print(y.shape)
# # pca = PCA(n_components=2)
# # X_r = pca.fit(X).transform(X)
# predict_cluster_label = KMeans(n_clusters=4).fit_predict(X_r)
# print(predict_cluster_label.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=10)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=10,max_depth=9,min_samples_split=7,min_samples_leaf=1) 
clf.fit(X_train, y_train)
y_test_pred = clf.predict(X_test)
acc = sum(y_test_pred == y_test) / len(y_test)
y_train_pred = clf.predict(X_train)  
train_acc = sum(y_train_pred == y_train) / len(y_train)  
print("Accuracy on training data: %.4f" % train_acc)
print("Accuracy on test data: %.4f" % acc)
print("Feature importances of input features:")
print(clf.feature_importances_)

cpdata = pd.read_csv("test_data_standard.csv", header=0, index_col=None)
Xcp = cpdata.iloc[:, 9:14].values
# X_r_r = pca.fit(X).transform(Xcp)
cy_test_pred = clf.predict(Xcp)
Xcp_with_label = np.concatenate((Xcp, cy_test_pred.reshape(-1, 1)), axis=1)
print(Xcp_with_label.shape)
dfXcp = pd.DataFrame(Xcp_with_label)
dfXcp.to_csv("no_load.csv",index=False)

