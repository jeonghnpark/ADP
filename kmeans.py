# 가져옴   https://github.com/Jihun-Dong/Machine-Learning-Practice/blob/5148be498cde8ce5f1737150a9301f1aac890946/%EA%B5%B0%EC%A7%91%ED%99%94.ipynb

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
iris.data
iris.feature_names
iris["feature_names"]

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
iris.target

# 군집의 개수 정하기

kmeans = KMeans(n_clusters=3)

kmeans.fit(df)

kmeans.labels_

type(kmeans.labels_)

df['target'] = iris.target

df.head()
df['cluster'] = kmeans.labels_

# groupby로 실제 군집과 예측 비교
result = df.groupby(by=['target', 'cluster']).count()[['sepal length (cm)']]

result

kmeans.cluster_centers_  # center 좌표?

df.loc[df['target'] == 0]
# 결과를 보니 cluster 1을 0으로 바꾸는게 좋겟다. 1->0,  0-> 1
df.loc[df['cluster'] == 0, 'cluster'] = 5
df.loc[df['cluster'] == 1, 'cluster'] = 0
df.loc[df['cluster'] == 5, 'cluster'] = 1
df.loc[df['target'] == 1]
df.loc[df['target'] == 2]  # 잘 안됨
# TODO target-pred비교하는 루틴

df['result'] = df['target'] - df['cluster']

print(f"accuracy is {df.loc[df['result'] == 0].shape[0] / df.shape[0]}")

# pca를 해서 차원을 축소해서 visualization 해보자
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transform = pca.fit_transform(iris.data)

df['pca_x'] = pca_transform[:, 0]
df['pca_y'] = pca_transform[:, 1]
df.head()
mark0 = df[df['cluster'] == 0].index
mark1 = df[df['cluster'] == 1].index
mark2 = df[df['cluster'] == 2].index

plt.scatter(x=df.loc[mark0, 'pca_x'], y=df.loc[mark0, 'pca_y'], marker='o')
plt.scatter(x=df.loc[mark1, 'pca_x'], y=df.loc[mark1, 'pca_y'], marker='s')
plt.scatter(x=df.loc[mark2, 'pca_x'], y=df.loc[mark2, 'pca_y'], marker='^')
plt.show()

# make_blobs()를 이용해서 데이터 생성후 군집해보기
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)

# dataframe 타입으로 변경
df = pd.DataFrame(X, columns=['ftr1', 'ftr2'])
df['target'] = y
df.head()

# visual

markers = ['o', 's', '^', 'P', 'D', 'H', 'x']
for target in list(np.unique(y)):
    target_cluster = df[df['target'] == target]
    plt.scatter(x=target_cluster['ftr1'], y=target_cluster['ftr2'], marker=markers[target])
plt.show()

# k-mean clustering 수행
kmeans = KMeans(n_clusters=3, max_iter=200, random_state=200)

# fit the model to the data
cluster_labels = kmeans.fit_predict(X)
df['label'] = cluster_labels

# central point
centers = kmeans.cluster_centers_

df.loc[df['label'] == 0]
# visual
markers = ['o', 's', '^', "P", "D", "H", 'x']
# for label in pd.unique(cluster_labels):
#     plt.scatter(x=df.loc[df['label'], ''] == label], y =)
