# 程序员：刘瑞翔
# 编译时间：2022/1/22 22:30
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
data[data[:,4] == 'Iris-versicolor',4] = -1
data_train,data_test = train_test_split(data,test_size=0.2)
#对超平面进行迭代逼近正确值,选用二维变量进行判断,准确率为1
w1,w2,b = 0,0,0#设置初始横纵坐标系数和截距
start = True
i = 0#迭代计数
while start:
    start = False
    i += 1
    for test in data_train:
        if test[4]*(test[1]*w2+test[0]*w1 + b )  <= 0:
            w1 = w1 +test[4]*test[0]
            w2 = w2 +test[4]*test[1]
            b = b +test[4]
            start = True
    if i == 500:
        start = False
x = np.linspace(4, 8, 50)
y = -(w1/w2)*x - b/w2

plt.rcParams['font.sans-serif'] = ['SimHei']##支持中文显示
plt.figure('')# 创建一个图像
plt.scatter(data_train[data_train[:,4] == 1,0], data_train[data_train[:,4] == 1,1], 15, 'b', marker='o',label='第一类训练集')
plt.scatter(data_train[data_train[:,4] == -1,0], data_train[data_train[:,4] == -1,1], 15, 'r', marker='o',label='第二类训练集')
plt.scatter(data_test[data_test[:,4] == 1,0], data_test[data_test[:,4] == 1,1], 15, 'b', marker='*',label='第一类测试集')
plt.scatter(data_test[data_test[:,4] == -1,0], data_test[data_test[:,4] == -1,1], 15, 'r', marker='*',label='第二类测试集')

plt.plot(x, y)
plt.title('萼片长度和宽度的种类分布图')
plt.xlabel('EpalLengthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.ylabel('SepalWidthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.legend(loc="best", fontsize=6)
plt.show()




