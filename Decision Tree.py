from math import log
import operator
import numpy as np


# 创建决策树的数据集和标签集
def createDataSet():
    dataset = np.array([[0, 0, 0, 0, 0],  # 数据集创立，最后一项0为no，1为yes
                        [0, 0, 0, 1, 0],
                        [0, 1, 0, 1, 1],
                        [0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 0, 1, 2, 1],
                        [1, 0, 1, 2, 1],
                        [2, 0, 1, 2, 1],
                        [2, 0, 1, 1, 1],
                        [2, 1, 0, 1, 1],
                        [2, 1, 0, 2, 1],
                        [2, 0, 0, 0, 0]])
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return dataset, labels


# 输出指定列的经验熵
def comEnt_empirical(data, axis):
    class_last = data[:, axis]  # 取出指定列
    data_len = len(data[:, axis])  # 得出一列含有几个数据
    class_label = set(class_last)  # 输出含有几个不同的类
    Ent_empirical = 0  # 定义经验熵变量
    for key in class_label:  # 对不同类别取出进行统计其个数
        contain = 0  # 记录个数容器
        for test in class_last:  # 遍历判断所有的数据归类数据
            if test == key:
                contain += 1
        prob = contain / data_len  # 得出每个分类的概率
        Ent_empirical += -(prob * log(prob, 2))  # 计算得出最后的经验熵
    return Ent_empirical


# 应用指定特征划分数据集，输出删除特征值后的符合给定值的数据
def splitdata(dataset, axis, value):
    dataset = dataset[dataset[:, axis] == value]  # 对数据集中指定列中等于value的行输出
    return np.delete(dataset, axis, axis=1)  # 对上式的输出的行删除指定列，并输出后的数组


# 计算并选择信息增益最大的特征
def chooes_Ent(dataset):
    D_Ent = comEnt_empirical(dataset, -1)  # 计算初始经验熵
    p_Ent = []  # 信息增益初始化
    for i in range(len(dataset[0]) - 1):  # 对每一个特征进行计算其条件经验熵
        len_axis1 = len(dataset)  # 统计数据的个数
        class_label = set(dataset[:, i])  # 得出第i个特征具有的类别
        key_Ent = 0  # 条件经验熵

        for key in class_label:  # 对每一个类别进行取出，统计其具有的个数
            contain = 0  # 统计同一类别数据个数容器
            for test in dataset[:, i]:  # 对数据集的指定特征取出，得出相应的类别
                if test == key:
                    contain += 1
            p_condition = contain / len_axis1  # 前提条件的概率
            new_data = splitdata(dataset, i, key)  # 得出符合条件的数据集
            key_Ent += p_condition * comEnt_empirical(new_data, -1)
        p_Ent.append(D_Ent - key_Ent)  # 每一个特征下的信息增益
    sort_Ent = sorted(range(len(p_Ent)), key=lambda k: p_Ent[k], reverse=True)
    return sort_Ent[0]  # 得出信息增益最大的特征


# 统计数据集中的最多的类
def comclass(dataset):
    class_data = set(dataset[:, -1])  # 得出指定列的类别
    label_class = []  # 存储set操作后标签种类
    num_class = []  # 存储每个类的个数
    for key in class_data:  # 对每一类进行判断
        label_class.append(key)
        contain = 0
        for test in dataset[:, -1]:
            if key == test:
                contain += 1
        num_class.append(contain)
    sort_p = sorted(range(len(num_class)), key=lambda k: num_class[k], reverse=True)
    return label_class[sort_p[0]]  # 输出该列数据中，拥有最多数据的类别


# 创建决策树
def creattree(dataset):
    if len(set(dataset[:, -1])) == 1:  # 如果标签最后只剩下一类返回树
        if dataset[0, -1] == 0:  # 对标签0，1还原为no，yes
            return 'no'
        else:
            return 'yes'
    if len(dataset[0]) == 1:
        if comclass(dataset) == 0:
            return 'no'
        else:
            return 'yes'

    featlabel = chooes_Ent(dataset)  # 信息增益最大的特征列名
    label = labels[featlabel]  # 信息增益最大的标签名字
    data_feat = dataset[:, featlabel]  # 每个数据信息增益最大的特征的值
    label_feat = set(data_feat)  # 对该特征的种类进行输出
    key_feat = []  # 记录特征下的种类

    for key in label_feat:
        key_feat.append(key)
    np.delete(dataset, featlabel, axis=1)  # 删除信息增益最大的特征
    tree = {label: {}}  # 利用字典嵌套字典创建树
    for value in key_feat:  # 递归建立树
        tree[label][value] = creattree(splitdata(dataset, featlabel, value))
    print(tree)
    return tree


def testtree(mytree, test, labels):
    now_d = next(iter(mytree))  # 得到当前的第一个结点,得到该结点对应的标签
    # 也可以用list改变输出，now_d = list(mytree.keys())，now_d[0]
    second_d = mytree[now_d]  # 打开下一个字典
    key_label = labels.index(now_d)  # 找到当前结点所对应的列的位置
    for key in second_d.keys():
        if key == test[key_label]:
            if isinstance(second_d[key], str):
                label_test = second_d[key]
            else:
                label_test = testtree(second_d[key], test, labels)
    return label_test


if __name__ == '__main__':
    dataset, labels = createDataSet()
    Tree = creattree(dataset)
    print(Tree)
    test = [0, 0, 1, 0]  # 给定一个测试数据
    result = testtree(Tree, test, labels)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')
