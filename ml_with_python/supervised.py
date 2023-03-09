from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
cancer.keys()
cancer.data.shape
cancer.target_names
cancer.target
{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
