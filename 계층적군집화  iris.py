import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
iris_data = iris.data
# 덴드로그램 그리기
from scipy.cluster import hierarchy

plt.figure(figsize=(10, 7))
plt.title('Dendrograms')
dend = hierarchy.dendrogram(hierarchy.linkage(iris_data, method='ward'))  # 메소드 변경 가능
plt.show()

from sklearn.cluster import AgglomerativeClustering

# 한번 군집이 결정되면 바뀌는 일이 없음

# 유클리디안 거리에 와드 연결은 각 군집의 개수가 비슷하게 나온다. 희소 행렬에서는 jaccard / average 사용(군집의 개수가 일정하진 않음)
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(iris_data)

plt.figure(figsize=(10, 7))
plt.scatter(iris_data[:, 0], iris_data[:, 1], c=cluster.labels_)

plt.show()

iris_df2 = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df2['cluster'] = cluster.labels_
iris_df2.head()

# 각 군집의 특징을 알 수 있음, 데이터프레임의 groupby().mean() 이용
print(f"각 군집내 평균값 {iris_df2.groupby(by='cluster').mean()}")
