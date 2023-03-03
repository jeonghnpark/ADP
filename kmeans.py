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

1 == 1

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
plt.title("After PCA")
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
pd.unique(cluster_labels)
for label in pd.unique(cluster_labels):
    label_cluster = df[df['label'] == label]
    centers_x_y = centers[label]
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], marker=markers[label], edgecolor='k')

    # 중심시각화
    plt.scatter(x=centers_x_y[0], y=centers_x_y[1], s=200, color='white', alpha=0.9, edgecolor='k',
                marker=markers[label])
    plt.scatter(x=centers_x_y[0], y=centers_x_y[1], s=70, color='k', edgecolor='k', marker='' % label)

plt.show()

df.groupby('target')['label'].value_counts()

# 군집평가
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

iris = load_iris()

iris.feature_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
df['label'] = kmeans.labels_
df.head()

# 모든 개별데이터에 실루엣 계수를 구함
score_sample = silhouette_samples(iris.data, df['label'])
score_sample.shape
df['s_coef'] = score_sample
avg_score = silhouette_score(iris.data, df['label'])
print(f"붓꽃 데이터 실루엣 스코어(0~1) (k=3 일때) {avg_score:.3f}")
print("붓꽃 kmean 군집별 실루엣 스코어, ")
print(f"{df.groupby('label')['s_coef'].mean()}")

# 최적 k를 찾는법 
best_K = 0
best_score = -1

for i in range(2, 6):
    kmeans = KMeans(n_clusters=i, random_state=123)
    kmeans.fit(df)
    avg_score = silhouette_score(df, df['label'])
    score_samples = silhouette_samples(df, df['label'])
    if avg_score > best_score:
        best_score = avg_score
        df['label'] = kmeans.labels_
        df['s_score'] = score_samples
        best_K = i

print(f"best k is {best_K}")
print(f"best score is {best_score:.3f}")
print(f"군집별 평균\n {df.groupby('label')['s_score'].mean()}")

from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.7, random_state=0)

df = pd.DataFrame(X, columns=['ftr1', 'ftr2'])

df['target'] = y

best_bandwidth = estimate_bandwidth(X)

meanshift = MeanShift(bandwidth=best_bandwidth)
labels = meanshift.fit_predict(X)
center_position = meanshift.cluster_centers_

print(f"cluster label 유형 : {np.unique(labels)} \n best bandwidth ={best_bandwidth:.3f}")

# target과 estimat를 비교한다.
markers = ['o', 's', '^', 'P', "D", 'H', 'x']
df['label'] = labels

print(f"average silhouette_score is {silhouette_score(X, df['label'])}")

for label in np.unique(df['label']):
    label_cluster = df[df['label'] == label]
    center = center_position[label]
    plt.scatter(x=label_cluster['ftr1'], y=label_cluster['ftr2'], marker=markers[label])
    plt.scatter(x=center[0], y=center[1], s=100, marker='*')

from sklearn.mixture import GaussianMixture

iris = load_iris()
X = iris.data

lowest_bic = np.infty
bic = []

n_component_range = range(1, 7)

for n_component in n_component_range:
    gmm = GaussianMixture(n_components=n_component)
    gmm.fit(X)
    bic.append(gmm.bic(X))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

gmm = best_gmm
labels = gmm.predict(X)

print(f"average silhouette_score is {silhouette_score(X, labels)}")

plt.scatter(x=X[:, 0], y=X[:, 1], c=labels, s=40, cmap='viridis')
plt.title("cluster analysis for iris data using GMM")
plt.show()
