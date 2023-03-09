import matplotlib.pyplot as plt
import numpy as np

unique_labels = 5

colors = [plt.get_cmap('Spectral')(each) for each in np.linspace(0, 1, 5)]
# colors = plt.get_cmap('Spectral')(np.linspace(0, 1, 10))
a = np.linspace(0, 1, 5)

print(a)
