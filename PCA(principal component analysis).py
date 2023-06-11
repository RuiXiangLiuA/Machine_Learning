# 程序员：刘瑞翔
# 编译时间：2022/1/31 17:19
import numpy as np
import pandas as pd
from numpy.linalg import eig
from pylab import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# iris = load_iris()
data = pd.read_csv('Iris.csv')
data = np.array(data)
X = data[:,1:5].astype(np.float64)
k = 2

X = X - X.mean(axis = 0)                #向量X中心化，即使得x的均值为0
X_cov = np.cov(X.T, ddof = 0)           #计算向量X的协方差矩阵
eigenvalues,eigenvectors = eig(X_cov)   #计算协方差矩阵的特征值和特征向量
klarge_index = ~eigenvalues.argsort()[0:k] #排序后选取最大的K个特征值及其特征向量
k_eigenvectors = eigenvectors[klarge_index]
results =  np.dot(X, k_eigenvectors.T)     #用X与特征向量相乘
# print(results)
tot = sum(eigenvalues)#对四个特征值求和
cum_var_exp = np.array([(i/tot) for i in sorted(eigenvalues, reverse = True)])#分别求取
plt.bar(range(1,5), cum_var_exp, alpha = 0.5, align = 'center', label = 'individual var')
plt.step(range(1,5), cum_var_exp, where = 'mid', label = 'cumulative var')
plt.ylabel('variance rtion')
plt.xlabel('principal components')
plt.legend(loc = 'best')
plt.show(block=True)








