import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# 3개의 중심을 가진 합성 데이터셋 생성
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.6)

# MeanShift 알고리즘의 대역폭 추정
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# MeanShift 알고리즘 인스턴스 생성
ms = MeanShift(bandwidth=bandwidth)

# 데이터에 알고리즘 적용
ms.fit(X)

# 클러스터 및 알고리즘이 찾은 중심 그래프에 표시
# Plot the clusters and the centers found by the algorithm
labels = ms.labels_
cluster_centers = ms.cluster_centers_

n_clusters_ = len(np.unique(labels))

plt.figure(1)
plt.clf()

colors = list('bgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
