# ===================================================================================
# knn的遍历算法与构建kd树方法，并应用到鸢尾花数据集
# ===================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from math import sqrt
from collections import namedtuple

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示中文
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列
# ------------------------------------------------------------------------------------

# 遍历算法
class KNN(object):

    def __init__(self,X_train, y_train, X_test, y_test, k, p):   # 训练集、测试集、临近点个数、范数距离
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.k = k
        self.p = p

    # 距离函数，输入待求距离的数组，输出距离
    def distance(self, x, y):  # 一维列向量
        sum = 0
        for i in range(len(x)):
            sum = sum+math.pow(abs(x[i] - y[i]), self.p)
        L = math.pow(sum,1/self.p)
        return L

    # 模型函数,输入待分类数据x，输出x所属于的类
    def model(self, x):
        L = np.zeros(len(self.X_train))

        for i in range(len(self.X_train)):
            L[i] = self.distance(x, self.X_train[i])

        L_location = np.argsort(L)   # 返回排序位置
        L_location_k = L_location[0:self.k]  # 获取k个临近点的位置
        L_label_k = self.y_train[L_location_k]   # k个临近点的分类结果
        classify = max(set(L_label_k.tolist()), key=L_label_k.tolist().count)  # 最终分类结果

        return classify

    # 预测函数，返回预测结果
    def predict(self):
        y_prediction = np.zeros(len(self.X_test))

        for i in range(len(self.X_test)):
            y_prediction[i] = self.model(self.X_test[i])

        return y_prediction

    #预测结果可视化
    def visualization(self):

        y_prediction = self.predict()
        test_matrix = confusion_matrix(self.y_test, y_prediction)  # 混淆矩阵

        print(" ")
        print("-----------------------------遍历算法预测结果-----------------------------")
        print("混淆矩阵：")
        print(test_matrix)
        accuracy = (test_matrix[0,0]+test_matrix[1,1])/sum(test_matrix).sum()
        print("准确率：",accuracy)

        df_train = pd.DataFrame(self.X_train)  # 将训练集转化为数据框
        df_train['label'] = self.y_train  # 新增标签列
        df_train.columns = ['sepal length', 'sepal width','label']  # 特征名称
        df_train_label0 = df_train.where(df_train.label == 0)
        df_train_label1 = df_train.where(df_train.label == 1)

        df_test_prediction = pd.DataFrame(self.X_test)  # 将训练集转化为数据框
        df_test_prediction['label'] = y_prediction  # 新增标签列
        df_test_prediction.columns = ['sepal length', 'sepal width', 'label']  # 特征名称
        df_test_prediction_label0 = df_test_prediction.where(df_test_prediction.label == 0)
        df_test_prediction_label1 = df_test_prediction.where(df_test_prediction.label == 1)

        plt.figure(figsize=(8, 8))
        plt.plot(df_train_label0['sepal length'], df_train_label0['sepal width'], '*r', label="训练集标签=0")
        plt.plot(df_train_label1['sepal length'], df_train_label1['sepal width'], '*g', label="训练集标签=1")
        plt.plot(df_test_prediction_label0['sepal length'], df_test_prediction_label0['sepal width'], 'or', label="测试集标签=0")
        plt.plot(df_test_prediction_label1['sepal length'], df_test_prediction_label1['sepal width'], 'og', label="测试集标签=1")
        plt.xlabel("sepal length", fontsize=18)
        plt.ylabel("sepal width", fontsize=18)
        plt.title("遍历算法---鸢尾花数据集测试结果", fontsize=15)
        plt.legend()
        plt.show(block=True)

# 数据集载入与整理
iris = load_iris()   # 载入数据集
df = pd.DataFrame(iris.data,columns=iris.feature_names)  # 将特征转化为数据框
df['label'] = iris.target   # 新增标签列
df.columns = ['sepal length','sepal width','petal length','etal width','label']  # 特征名称

print("------------------------------数据集信息----------------------------------")
print(df.head())  # 查看前五行
print(df.describe())  # 查看数据集基本信息

# 取出标签为0和1的数据进行可视化
iris_label0 = df.where(df.label==0)
iris_label1 = df.where(df.label==1)

plt.figure(figsize=(8,8))
plt.plot(iris_label0['sepal length'],iris_label0['sepal width'],'or',label="0")
plt.plot(iris_label1['sepal length'],iris_label1['sepal width'],'og',label="1")
plt.xlabel("sepal length",fontsize = 18)
plt.ylabel("sepal width",fontsize = 18)
plt.title("鸢尾花部分数据集",fontsize = 15)
plt.legend()

# 训练集与测试集的划分
data = np.array(df.iloc[:100,[0,1,4]])
X = data[:,:2]
y = data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# 模型训练与测试
k = 3  # 临近点个数
p = 2  # 距离
clf = KNN(X_train, y_train, X_test, y_test, k, p)   # 模型
clf.visualization()  # 数据可视化
# ------------------------------------------------------------------------------------

# scikit-learn实现
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
y_prediction = clf_sk.predict(X_test)
test_matrix = confusion_matrix(y_test, y_prediction)  # 混淆矩阵

print(" ")
print("-----------------------------scikit-learn预测结果-----------------------------")
print("混淆矩阵：")
print(test_matrix)
accuracy = (test_matrix[0,0]+test_matrix[1,1])/sum(test_matrix).sum()
print("准确率：",accuracy)
# ------------------------------------------------------------------------------------


# kd树算法
class KdNode(object):

    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # 节点
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree

class KdTree(object):

    def __init__(self, data):
        k = len(data[0])  # 数据维度

        def CreateNode(split, data_set):  # 按第split维划分数据集创建KdNode
            if not data_set:  # 数据集为空
                return None
            data_set.sort(key=lambda x: x[split])  # 按要分割的那一维度排序
            split_pos = len(data_set) // 2  # //为Python中的整数除法，数据个数
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # 下一分割维度

            # 递归的创建kd树
            return KdNode(
                median,
                split,
                CreateNode(split_next, data_set[:split_pos]),  # 创建左子树
                CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, data)  # 从第0维分量开始构建kd树,返回根节点

# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)

# 对构建好的kd树进行搜索，寻找与目标点最近的样本点
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数

def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"),
                          0)  # python中用float("inf")和float("-inf")表示正负无穷

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的中值点

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited = nodes_visited+temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        temp_dist = sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pivot, target))) # 计算目标点与分割点的欧氏距离

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        temp2 = travel(further_node, target, max_dist) # 检查另一个子结点对应的区域是否有更近的点

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归

#应用到鸢尾花数据集
kd = KdTree(X_train.tolist())
print(" ")
print("------------------------鸢尾花数据训练集根节点------------------------")
preorder(kd.root)

#预测
X_test_nearest = np.empty(X_test.shape)  # 最近点
y_prediction = np.empty(y_test.shape)    # 预测值

for i in range(len(X_test)):
    X_test_nearest[i] = find_nearest(kd, X_test[i]).nearest_point    # 最近点
    y_index = X_train.tolist().index(X_test_nearest[i].tolist())     # 最近点索引
    y_prediction[i] = y_train[y_index]  # 预测值

#可视化
test_matrix = confusion_matrix(y_test, y_prediction)  # 混淆矩阵

print(" ")
print("-----------------------------kd树预测结果-----------------------------")
print("混淆矩阵：")
print(test_matrix)
accuracy = (test_matrix[0,0]+test_matrix[1,1])/sum(test_matrix).sum()
print("准确率：",accuracy)

df_train = pd.DataFrame(X_train)  # 将训练集转化为数据框
df_train['label'] = y_train  # 新增标签列
df_train.columns = ['sepal length', 'sepal width','label']  # 特征名称
df_train_label0 = df_train.where(df_train.label == 0)
df_train_label1 = df_train.where(df_train.label == 1)

df_test_prediction = pd.DataFrame(X_test)  # 将训练集转化为数据框
df_test_prediction['label'] = y_prediction  # 新增标签列
df_test_prediction.columns = ['sepal length', 'sepal width', 'label']  # 特征名称
df_test_prediction_label0 = df_test_prediction.where(df_test_prediction.label == 0)
df_test_prediction_label1 = df_test_prediction.where(df_test_prediction.label == 1)

plt.figure(figsize=(8, 8))
plt.plot(df_train_label0['sepal length'], df_train_label0['sepal width'], '*r', label="训练集标签=0")
plt.plot(df_train_label1['sepal length'], df_train_label1['sepal width'], '*g', label="训练集标签=1")
plt.plot(df_test_prediction_label0['sepal length'], df_test_prediction_label0['sepal width'], 'or', label="测试集标签=0")
plt.plot(df_test_prediction_label1['sepal length'], df_test_prediction_label1['sepal width'], 'og', label="测试集标签=1")
plt.xlabel("sepal length", fontsize=18)
plt.ylabel("sepal width", fontsize=18)
plt.title("kd树算法---鸢尾花数据集测试结果", fontsize=15)
plt.legend()
plt.show(block=True)

