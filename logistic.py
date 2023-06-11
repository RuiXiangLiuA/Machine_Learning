import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pylab import *
import matplotlib.pyplot as plt
#导入数据集
data = pd.read_csv('Iris.csv')
del data['Id']
data = np.array(data.iloc[0:100,:])#用数组存储数据点
data[data[:,4] == 'Iris-setosa',4] = 1
data[data[:,4] == 'Iris-versicolor',4] = 0
data = np.insert(data,2,np.full((1,100),1),axis=1)
data_train,data_test = train_test_split(data,test_size=0.2)

k = 1000#迭代次数
alpha = 0.1#参数

def sigmoid(w, x):  # sigmoid函数，输入参数，数据集，返回概率
    result = np.exp(x @ w) / (1 + np.exp(x @ w))
    return result

n = 3
w = np.ones(n)
for p in range(k):
    for j in range(n):
        gradient = np.inner(data_train[:,5] - sigmoid(w, data_train[:,0:3].astype(np.float64)), data_train[:, j])  # 梯度
        w[j] = w[j] + alpha * gradient  # 更新

x = np.linspace(4, 8, 50)
y = -(w[0]/w[1])*x-(w[2]/w[1])

plt.rcParams['font.sans-serif'] = ['SimHei']##支持中文显示
plt.figure('')# 创建一个图像
plt.scatter(data_train[data_train[:,5] == 1,0], data_train[data_train[:,5] == 1,1], 15, 'b', marker='o',label='第一类训练集')
plt.scatter(data_train[data_train[:,5] == 0,0], data_train[data_train[:,5] == 0,1], 15, 'r', marker='o',label='第二类训练集')
plt.scatter(data_test[data_test[:,5] == 1,0], data_test[data_test[:,5] == 1,1], 15, 'b', marker='*',label='第一类测试集')
plt.scatter(data_test[data_test[:,5] == 0,0], data_test[data_test[:,5] == 0,1], 15, 'r', marker='*',label='第二类测试集')

plt.plot(x, y)
plt.title('萼片长度和宽度的种类分布图')
plt.xlabel('EpalLengthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.ylabel('SepalWidthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.legend(loc="best", fontsize=6)
plt.show()