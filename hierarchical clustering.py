# 程序员：刘瑞翔
# 编译时间：2022/1/15 9:46
from pylab import *
import numpy as np
import pandas as pd
import copy
#导入数据集
data = pd.read_csv('Iris.csv')
del data['Id']
del data['Species']
data = np.array(data)#用数组存储数据点
re_data = copy.deepcopy(data)#用于判断距离的临时存储数组

#距离函数
def distances(follow_1,follow_2):
    '''求取两个数据点的欧式距离'''
    if follow_2 is None or follow_1 is None:#将空值与各点的距离剔除
        return None
    else:
        return np.linalg.norm(follow_1-follow_2)

#平均值函数
def average(species):
    '''对类进行平求均值更新中心点'''
    axis_0 = 0
    for species_test in species[:,0]:
        if  species_test != None:
            axis_0 =  axis_0 + 1
        if species_test == None:
            break
    list_center = []
    for j in range(4):
        if axis_0 == 0:
            list_center.append(0)
        else:
            sum1 = 0
            for i in range(axis_0):
                sum1 = sum1 + species[i,j]
            list_center.append(sum1/axis_0)
    p =  np.array(list_center)
    return p

show_label = np.full((150,150),None)#以行坐标展示层次次数，列坐标展示点
length_data = len(data)#将数据个数求出
#创建二维数组将150个点分别存储在第一行
con_data = np.full((150,150),None)
for i in range(length_data):
    con_data[0,i] = i

#对聚类进行多次层次聚类
for times in range(148):
    for every_data in range(150):#将点分类
        for i,axis_data in enumerate(con_data):
            if every_data in axis_data:
                every_index = axis_data.tolist().index(every_data)#得到每个值在con_data中的列坐标
                show_label[every_data,times] = every_index#对数据点的结果代入
    contain = np.full((149, 149), None, dtype=float)  # 创建一个149行149列的全零数组准备存储距离
    for i in range(length_data-1):#遍历所有数据点进行距离求值
        follow_1 = re_data[[i]]
        for j in range(i+1,length_data):
            follow_2 = re_data[[j]]
            contain[i,j-1] = distances(follow_1,follow_2)

    #对求出的距离进行排序
    test =  contain.flatten()#将所得的距离二维数组转化为一维数组
    sorted = np.argsort(test)#排序
    x = (sorted[0])//149  #求出距离最近对应的两点
    y = (sorted[0])%149+1

    #将聚类成功的点存入新的类中
    for middle in con_data[:,y]:
            if middle != None:
                for i in range(length_data):
                    if con_data[i,x] == None:
                        con_data[i,x] = middle
                        break   #只将行数最小的那个数据进行赋值

    #将并入的新类求出平均值，删除旧类的数据
    species = np.full((150,4),None)
    i  = 0                          #用于标记合并类的位置
    for index in con_data[:,x]:
        if index != None:           #对非空值进行操作
            species[[i]] = data[[index]]
            i = i + 1
    n_average = average(species)    #新类的中心点

    for y_test in con_data[:,y]:    #循环取出旧类中数据编号，对re_data数据中的原先类删除
        if y_test != None:
            re_data[[y_test]] = None
    con_data[:,y] = None            #对旧类赋空值
    re_data[[x]] = n_average        #将新类的中心点坐标更新


#将所得类的数据点得出
last_species = []   #存储类的首个序号
con_species = [[] for i in range(3)]#存储类的个数
j  = 0  #记录类的序号
for w_species in con_data[0,:]:#取出分类后的类进行判断类的位置
    if w_species != None:
        last_species.append(w_species)#记录类的首位置
        k = 0 #记录类的行数
        for point in con_data[:,w_species]:#取出每一类的数据点，判断出具有实位置的点
            if point is not None:
                con_species[j].append(point)
                k += 1
        j += 1

species_0 = pd.DataFrame(data[[con_species[0]]])
species_1 = pd.DataFrame(data[[con_species[1]]])
species_2 = pd.DataFrame(data[[con_species[2]]])

plt.rcParams['font.sans-serif'] = ['SimHei']##支持中文显示
plt.figure('')# 创建一个图像
plt.scatter(species_0[0], species_0[1], 15, 'r', marker='o', label='第一类')
plt.scatter(species_1[0], species_1[1], 15, 'y', marker='o',label='第二类')
plt.scatter(species_2[0], species_2[1], 15, 'b', marker='o',label='第三类')
plt.title('萼片长度和宽度的种类分布图')
plt.xlabel('EpalLengthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.ylabel('SepalWidthCm', fontdict={'family': 'Times New Roman', 'size': 13})
plt.legend(loc="best", fontsize=6)
plt.show(block=True)