# ===================================================================================
# 线性svm的原始形式与对偶形式
# ===================================================================================
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示中文
pd.set_option('display.max_rows', None)  # 显示全部行
pd.set_option('display.max_columns', None)  # 显示全部列

# 可视化，输入数据与超平面，输出图片
def visualization(x,y,w):

    x1 = np.arange(0, 6, 0.1)
    x2 = (-w[0] * x1 - w[2]) / w[1]
    positive = np.where(y == 1)  # 正类
    negative = np.where(y == -1)  # 负类

    plt.figure(figsize=(8, 8))
    plt.plot(x1,x2,color="blue",label='分类超平面')
    plt.plot(x[positive][:,0],x[positive][:,1],"or",label='正类')
    plt.plot(x[negative][:, 0], x[negative][:, 1], "og", label='负类')
    plt.xlabel("X",fontsize=18)
    plt.ylabel("Y", fontsize=18)
    plt.title("分类结果",fontsize=18)
    plt.legend()
# -------------------------------------------------------------------------

# 例7.1,感知机算法原始形式
x = np.array([[3,3],[4,3],[1,1]])  # 数据
y = np.array([1,1,-1])    # 标签
fun = lambda w: ((w[0]) ** 2 + (w[1]) ** 2)/2
cons = ({'type': 'ineq', 'fun': lambda w: y[0]*(x[0,0] * w[0] + x[0,1] * w[1] + w[2]) - 1},
        {'type': 'ineq', 'fun': lambda w: y[1]*(x[1,0] * w[0] + x[1,1] * w[1] + w[2]) - 1},
        {'type': 'ineq', 'fun': lambda w: y[2]*(x[2,0] * w[0] + x[2,1] * w[1] + w[2]) - 1})
res = optimize.minimize(fun, np.ones(3), method='SLSQP', constraints=cons)
print("------------------------原始形式优化结果------------------------")
print(res)
w = res.x  #分类超平面
visualization(x,y,w)
# -------------------------------------------------------------------------


# 例7.2，感知机算法对偶形式
def fun(a):   # 优化目标
    fx = 4 * (a[0]) ** 2 + 13 / 2 * (a[1]) ** 2 + 10 * a[0] * a[1] - 2 * a[0] - 2 * a[1]
    return fx

bnds = ((0, None), (0, None))  #a大于等于0
res = optimize.minimize(fun, np.ones(2), method='SLSQP', bounds=bnds)
alpha = res.x
alpha = np.append(alpha, alpha[0] + alpha[1])  #alpha值
w = np.sum((alpha * y).reshape(-1, 1) * x, axis=0)  #w
j = np.argmax(alpha)   #选择最大的正分量
b = y[j] - np.sum(alpha * y * np.dot(x, x[j, :]), axis=0)
w = np.append(w,b)
print("------------------------对偶形式优化结果------------------------")
print(w)
visualization(x,y,w)