# 编译时间：2022/1/27 12:25
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv('boston_housing_data.csv')#读入数据
plt.figure(figsize=(8,4))#创建图像
plt.scatter(data['CRIM'],data['MEDV'],c = 'red')#创建散点图
plt.xlabel('CRIM')
plt.ylabel('MEDV')
plt.show(block=True)


data = data[(~data.MEDV.isnull())]#删除数据框MEDV列中空值的行，
#isnull函数是判断数据组中是否含有空值，~表示取反操作。
#data.dropna(subset=["MEDV"],axis=0,inplace=True)#利用dropna函数也可以剔除空值
x = data.drop(['MEDV'],axis = 1)
y = data[['MEDV']]#一个括号输出数据表，两个括号输出数据框
#print(data.isnull().any())#测试数据集中是否含有空值
reg = LinearRegression()
reg.fit(x,y)
X = np.array(data.drop(['MEDV'],axis = 1))
y = data['MEDV']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())










