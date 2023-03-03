import matplotlib.pyplot as plt
import numpy as np

# 데이터 생성
x = np.random.randn(100)
y = np.random.randn(100)

# 산점도 그리기
plt.scatter(x, y, marker='o')  # 동그라미
plt.scatter(x, y + 1, marker='s')  # 사각형
plt.scatter(x, y + 2, marker='^')  # 세모

# 그래프 보여주기
plt.show()
