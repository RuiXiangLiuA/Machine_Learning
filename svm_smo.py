# ===================================================================================
# 求解svm的smo算法，并应用到鸢尾花数据集
# ===================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示中文
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列

class svm_smo:

    def __init__(self, X_train, y_train, alpha, C, iteration, kernel):   # 训练集、初值、松弛变量、迭代次数、核函数
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.b = 0   # 初始化偏置为0
        self.iteration = iteration
        self.kernel = kernel
        self.m = X_train.shape[0]   # 训练集个数
        self.n = X_train.shape[1]   # 训练集维度
        self.C = C
        self.E = [self.Ei(i) for i in range(self.m)]  #E

    def kernel_function(self,x,y):  # 核函数

        if self.kernel =='linear':
            result = np.inner(x,y)

        return result

    def g(self,i):  # xi的预测值
        gx = sum(self.alpha[j] * self.y_train[j] * self.kernel_function(self.X_train[i], self.X_train[j]) for j in range(self.m))+self.b
        return gx

    def Ei(self,i):   # Ei,预测值与真实值之差
        result = self.g(i)-self.y_train[i]
        return result

    def KKT(self,i):    # 判断是否满足KKT条件

        yi_gxi = self.y_train[i]*self.g(i)
        if self.alpha[i]==0:
            return  yi_gxi>=1
        elif 0<self.alpha[i]<self.C:
            return yi_gxi==1
        else:
            return yi_gxi<=1

    def choose_alpha(self):   #选择下一次更新的alpha

        satisfy_index = []
        for k in range(self.m):
            if 0 < self.alpha[k] < self.C:
                satisfy_index.append(k)  # 满足0<alpha<c的点的索引

        nonsatisfy_index = []
        for k in range(self.m):
            if k not in satisfy_index:
                nonsatisfy_index.append(k)  # 不满足0<alpha<c条件的点的索引

        satisfy_index.extend(nonsatisfy_index)

        for i in satisfy_index:

            if self.KKT(i):
                continue

            E1 = self.E[i]
            if E1 >= 0:  # 根据E1选择E2
                j = np.argmin(self.E)
            else:
                j = np.argmax(self.E)

            return i,j


    def bound(self,i,j):  # 确定边界

        if self.y_train[i] == self.y_train[j]:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])
        else:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

        return L,H

    def compare(self,alpha2_new_unc,L,H):  # alpha2剪枝

        if alpha2_new_unc>H:
            alpha2_new = H
        elif L<=alpha2_new_unc<=H:
            alpha2_new = alpha2_new_unc
        else:
            alpha2_new = L

        return alpha2_new

    def model(self):  # 模型

        print("---------------------------训练中---------------------------")

        for p in range(self.iteration):

            i,j = self.choose_alpha()   #选择更新的alpha索引
            E1 = self.Ei(i)
            E2 = self.Ei(j)
            eta = self.kernel_function(self.X_train[i,], self.X_train[i,]) + \
                  self.kernel_function(self.X_train[j,], self.X_train[j,])- \
                  2*self.kernel_function(self.X_train[i,], self.X_train[j,])

            #计算需更新的值
            alpha2_new_unc = self.alpha[j] + self.y_train[j]*(E1-E2)/eta   #未剪枝的alpha2
            L,H = self.bound(i,j)
            alpha2_new = self.compare(alpha2_new_unc, L, H)   #剪枝后的alpha2
            alpha1_new = self.alpha[i] + self.y_train[i] * self.y_train[j] * (self.alpha[j]-alpha2_new)
            b1_new = -E1 - self.y_train[i] * self.kernel_function(self.X_train[i,],self.X_train[i,]) * \
                     (alpha1_new - self.alpha[i]) - self.y_train[j] * self.kernel_function(self.X_train[j,],self.X_train[i,])* \
                     (alpha2_new - self.alpha[j]) + self.b
            b2_new = -E2 - self.y_train[i] * self.kernel_function(self.X_train[i,],self.X_train[j,]) * \
                     (alpha1_new - self.alpha[i]) - self.y_train[j] * self.kernel_function(self.X_train[j,],self.X_train[j,])* \
                     (alpha2_new - self.alpha[j]) + self.b

            if 0 < alpha1_new < self.C & 0 < alpha2_new < self.C:
                b_new = b1_new
            else:
                b_new = (b1_new+b2_new)/2

            #参数更新
            self.alpha[i] = alpha1_new
            self.alpha[j] = alpha2_new
            self.b = b_new

            self.E[i] = self.Ei(i)
            self.E[j] = self.Ei(j)

            print("已迭代次数： %f"%p)

            if p==self.iteration-1:

                print("训练结束！")

                self.w = np.dot((self.y_train.reshape(-1, 1) * self.X_train).T,self.alpha)
                # jj = np.where(self.alpha>0)[0][1]
                # self.b = self.y_train[jj]-sum(self.alpha[i]*self.y_train[i]*np.inner(self.X_train[i], self.X_train[jj]) for i in range(self.m))


    def predict(self,X_test):
        y_prediction = np.zeros(X_test.shape[0])

        for i in range(X_test.shape[0]):

            result = np.inner(X_test[i],self.w)+self.b

            if result>0:
                y_prediction[i] = 1
            else:
                y_prediction[i] = -1

        return y_prediction



def visualization(y_test,y_prediction,w,b, name):   #可视化函数

    test_matrix = confusion_matrix(y_test,y_prediction)
    print("------------------------%s------------------------"%name)
    print("混淆矩阵:")
    print(test_matrix)
    accuracy = (test_matrix[0, 0] + test_matrix[1, 1]) / sum(test_matrix).sum()
    print("准确率:", accuracy)

    df_test_prediction = pd.DataFrame(X_test)  # 将训练集转化为数据框
    df_test_prediction['label'] = y_prediction # 新增标签列
    df_test_prediction.columns = ['sepal length', 'sepal width', 'label'] # 特征名称
    df_test_prediction_label0 = df_test_prediction.where(df_test_prediction.label == -1)
    df_test_prediction_label1 = df_test_prediction.where(df_test_prediction.label == 1)
    x1 = np.arange(4,7,0.1)
    x2 = (-w[0]*x1-b)/w[1]

    plt.figure(figsize=(8, 8))
    df_train = pd.DataFrame(X_train)  # 将训练集转化为数据框
    df_train['label'] = y_train  # 新增标签列
    df_train.columns = ['sepal length', 'sepal width', 'label']  # 特征名称
    df_train_label0 = df_train.where(df_train.label == -1)
    df_train_label1 = df_train.where(df_train.label == 1)

    if name=="自编svm算法测试结果":
        plt.plot(x1,x2,color='blue',label='分类超平面')

    plt.plot(df_train_label0['sepal length'], df_train_label0['sepal width'], '*r', label="训练集标签=-1")
    plt.plot(df_train_label1['sepal length'], df_train_label1['sepal width'], '*g', label="训练集标签=1")
    plt.plot(df_test_prediction_label0['sepal length'], df_test_prediction_label0['sepal width'], 'or', label="测试集标签=-1")
    plt.plot(df_test_prediction_label1['sepal length'], df_test_prediction_label1['sepal width'], 'og', label="测试集标签=1")
    plt.xlabel("sepal length", fontsize=18)
    plt.ylabel("sepal width", fontsize=18)
    plt.title(name, fontsize=15)

    plt.legend()



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
plt.plot(iris_label0['sepal length'],iris_label0['sepal width'],'or',label="-1")
plt.plot(iris_label1['sepal length'],iris_label1['sepal width'],'og',label="1")
plt.xlabel("sepal length",fontsize = 18)
plt.ylabel("sepal width",fontsize = 18)
plt.title("鸢尾花部分数据集",fontsize = 15)
plt.legend()

# 训练集与测试集的划分
data = np.array(df.iloc[:100,[0,1,4]])
X = data[:,:2]
y = data[:,-1]

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1   #将标签为0的转化为-1

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#模型训练
svm = svm_smo( X_train, y_train, np.ones(X_train.shape[0]), 1, 1000, 'linear')  # 初值为1，松弛变量为1，最大迭代次数为1000，核函数为线性核
svm.model()

#模型预测
y_prediction = svm.predict(X_test)

#模型可视化
visualization(y_test, y_prediction, svm.w, svm.b,"自编svm算法测试结果")
# --------------------------------------------------------------------------------------------


# 利用sklearn包进行求解
sklearn_svm = SVC(kernel='linear')
sklearn_svm.fit(X_train, y_train)
y_prediction = sklearn_svm.predict(X_test)
visualization(y_test, y_prediction, svm.w, svm.b,"sklearn包svm算法测试结果")