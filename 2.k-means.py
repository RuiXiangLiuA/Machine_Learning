# 编译时间：2022/1/11 21:50
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from pylab import *

k = 3
data = pd.read_csv('Iris.csv')
del data['Id']
sorting = pd.DataFrame()
sorting = data[['SepalLengthCm']]#取第一个特征值作为初始中心点选取的数据
m = len(sorting)
roll = [i for i in range(0,m)]

for i in range(0,m):#对第一个特征值进行排序
    for j in range(0,m-i-1):
        if sorting.iloc[j,0] < sorting.iloc[j+1,0]:
            sorting.iloc[j,0],sorting.iloc[j+1,0] = sorting.iloc[j+1,0],sorting.iloc[j,0]
            roll[j],roll[j+1] = roll[j+1],roll[j]

index = m//(k+1)#中心间隔大小
center = pd.DataFrame()
v_center = []
for i in range(0,k):
    v_center.append(roll[index*(i+1)-1])
    a = data.iloc[v_center[i],:]
    center = center.append(data.iloc[v_center[i],:-1])

####上代码初始化中心点，下列代码进行迭代
def average(species):
    '''对单个类中心点进行求平均值更新中心点'''
    list_center = []
    for j in range(4):
        sum1 = 0
        for i in range(len(species)):
            sum1 = sum1 + species.iloc[i,j]
        list_center.append(sum1/len(species))
    p =  pd.DataFrame(list_center)
    return p.T

#定义距离函数
def distances(follow, center_x):
    return np.sqrt(sum((follow[:,0:-1] - center_x) ** 2))
for j in range(5):
    #更新三个数据框记录不同的类
    species_1 = pd.DataFrame()
    species_2 = pd.DataFrame()
    species_3 = pd.DataFrame()

    #将每一个点与中心的距离求出，并将其分类
    for follow in data.values:
        follow = follow.reshape((1,5))
        min_dis = 1000
        n_dis = np.array(np.zeros(k))
        center = np.array(center)
        for i in range(0,k):
            n_dis[i] = distances(follow,center[i])

        sorted_dis = np.argsort(n_dis)#将所得距离从小到大进行排序
        x_follow = pd.DataFrame(follow)#对点进行归类
        if sorted_dis[0] == 0:
           species_1 = species_1.append(x_follow)
        if sorted_dis[0] == 1:
           species_2 = species_2.append(x_follow)
        if sorted_dis[0] == 2:
           species_3 = species_3.append(x_follow)

    #更新中心点
    center = pd.DataFrame()
    center = center.append(average(species_1))
    center = center.append(average(species_2))
    center = center.append(average(species_3))
    #print(center)

plt.rcParams['font.sans-serif'] = ['SimHei']##支持中文显示
plt.figure('')# 创建一个图像
plt.scatter(species_1[0], species_1[1], 15, 'r', marker='o', label='第一类')
plt.scatter(species_2[0], species_2[1], 15, 'y', marker='o',label='第二类')
plt.scatter(species_3[0], species_3[1], 15, 'b', marker='o',label='第三类')
plt.legend(loc="best", fontsize=6)
plt.show(block=True)
