# ===================================================================================
# 利用GEM算法3求解高斯混合模型，并应用到鸢尾花数据集
# ===================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import random
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示中文
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列

class EM:

    def __init__(self, x, y, w,iteration): # x,y,初值,迭代次数
        self.x = x
        self.y = y
        self.w = w
        self.m = len(x)
        self.n = w.shape[0]
        self.iteration = iteration

    def gauss(self, x, thate, variance):   # 高斯分布密度函数
        prob = 1/((2*np.pi*variance)**(0.5))*np.exp(-(x-thate)**2/(2*variance))
        return prob

    def fit(self):

        gamma = np.empty([self.m, self.n])
        for p in range(self.iteration):

            for k in range(self.n):

                for j in range(self.m):

                    gamma[j,k] = self.w[k,0]*self.gauss(self.x[j],self.w[k,1],self.w[k,2])/\
                                 np.sum([self.w[i,0]*self.gauss(self.x[j],self.w[i,1],self.w[i,2]) for i in range(self.n)])
                self.w[k,1] = np.sum([gamma[i,k]*self.x[i] for i in range(self.m)])/\
                                  np.sum([gamma[i,k] for i in range(self.m)])
                self.w[k,2] = np.sum([gamma[i,k]*(self.x[i]-self.w[k,1])**2 for i in range(self.m)])/\
                                  np.sum([gamma[i,k] for i in range(self.m)])
                self.w[k,0] = np.sum([gamma[i,k] for i in range(self.m)])/self.m

            if (p==self.iteration-1):

                self.classification = gamma
                print("--------------------------------------最终迭代结果--------------------------------------")
                print("alpha1:%s   thate1:%s   variance1:%s"%(self.w[0,0],self.w[0,1],self.w[0,2]))
                print("alpha2:%s   thate2:%s   variance2:%s" % (self.w[1, 0], self.w[1, 1], self.w[1, 2]))

        self.classification = gamma

    def predict_label(self):

        if self.w[0,1]<self.w[1,1]:
            prediction = np.argmax(self.classification,axis=1)
        else:
            prediction = np.argmin(self.classification, axis=1)

        test_matrix = confusion_matrix(self.y, prediction)  # 混淆矩阵
        print(" ")
        print("--------------------------------------EM算法预测结果--------------------------------------")
        print("混淆矩阵：")
        print(test_matrix)
        accuracy = (test_matrix[0,0]+test_matrix[1,1])/sum(test_matrix).sum()
        print("准确率：",accuracy)

    def predict_unlabel(self):
        prediction = np.argmax(self.classification,axis=1)
        print("分类结果: ",prediction)
        return prediction

print("--------------------------------------鸢尾花数据集--------------------------------------")
# 数据集载入与整理
iris = load_iris()   # 载入数据集
df = pd.DataFrame(iris.data,columns=iris.feature_names)  # 将特征转化为数据框
df['label'] = iris.target   # 新增标签列
df.columns = ['sepal length','sepal width','petal length','etal width','label']  # 特征名称

# 取出标签为0和1的数据进行可视化
feature = 2 # 选择的特征
plt.figure(figsize=(8,8))
plt.plot(np.array(range(50)),df.iloc[0:50,feature],'or',label="0")
plt.plot(np.array(range(50)),df.iloc[50:100,feature],'og',label="1")
plt.xlabel("样本编号",fontsize = 18)
plt.ylabel("sepal length",fontsize = 18)
plt.title("鸢尾花部分数据集",fontsize = 15)
plt.legend()

#频率分布直方图
plt.figure(figsize=(8,8))
plt.hist(df.iloc[0:50,feature],alpha=0.8,label="0")
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
plt.xlabel('Sepal length',fontsize=18)
plt.ylabel('Number of Data',fontsize=18)
plt.legend()
plt.hist(df.iloc[50:100,feature],alpha=0.8,label="1")
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看
plt.xlabel('Sepal length',fontsize=18)
plt.ylabel('Number of Data',fontsize=18)
plt.title('Sepal length',fontsize=18)
plt.legend()
plt.show()

# 选取数据
sample = 100
index = [i for i in range(sample)]
random.shuffle(index)   # 打乱顺序
data = np.array(df.iloc[:sample,[feature,4]])
x = data[index,0]
y = data[index,-1]

# 模型训练
w = np.array([[0.5,1,1],[0.5,1,1]])  # 初值
iteration = 1000
model = EM(x, y, w, iteration)
model.fit()
model.predict_label()



# ===================================================================================
# 习题9.3
# ===================================================================================
print(" ")
print("----------------------------------------习题9.3---------------------------------------")
data1 = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60,75])

plt.figure(figsize=(8,8))
sequence = np.array(range(len(data1)))
plt.scatter(sequence,data1)
plt.xlabel("样本编号",fontsize = 18)
plt.ylabel("数值",fontsize = 18)
plt.title("数据集散点图",fontsize = 18)
plt.show()


model1 = EM(data1, [], np.array([[0.5,20,100],[0.5,-20,50]]), iteration)
model1.fit()
prediction = model1.predict_unlabel()  # 预测类别

label0 = np.where(prediction==0)
label1 = np.where(prediction==1)
plt.figure(figsize=(8,8))
plt.plot(sequence[label0],data1[label0],'or',label='0')
plt.plot(sequence[label1],data1[label1],'og',label='1')
plt.xlabel("样本编号",fontsize = 18)
plt.ylabel("数值",fontsize = 18)
plt.title("数据集分类散点图",fontsize = 18)
plt.legend()
plt.show()