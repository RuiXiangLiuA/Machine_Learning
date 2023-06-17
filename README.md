该项目为编译者大学期间自学机器学习算法，以解决数学建模及科研实践应用所学。

同时上传了一个自己处理的数据分析项目：《地震断层识别》
基于RANSAC点云多平面拟合分割和区域生长图像分割算法，实现地震断层识别

为力求自学算法精髓，代码仅使用如numpy等基础库。

参考书籍《统计学习方法》-李航，《机器学习》-周志华。

核心代码均含有详细注释，代码编译运行成功。
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

代码目录如下：

1.Knn:
KNN算法思想：对于任意n维输入向量，分别对应于特征空间中的一个点，输出为该特征向量所对应的类别标签或预测值。

2.k-means：
利用相似性度量方法来衡量数据集中所有数据之间的关系，将关系比较密切的数据划分到一个集合中。
（1）K-means算法首先需要选择K个初始化聚类中心
（2）计算每个数据对象到K个初始化聚类中心的距离，将数据对象分到距离聚类中心最近的那个数据集中，当所有数据对象都划分以后，
 就形成了K个数据集（即K个簇）
（3）接下来重新计算每个簇的数据对象的均值，将均值作为新的聚类中心
（4）最后计算每个数据对象到新的K个初始化聚类中心的距离，重新划分
（5）每次划分以后，都需要重新计算初始化聚类中心，一直重复这个过程，直到所有的数据对象无法更新到其他的数据集中。

3.PCA(principal component analysis):
即主成分分析方法，是一种使用最广泛的数据降维算法。PCA的主要思想是将n维特征映射到k维上，这k维是全新的正交特征也被称为主成分，
是在原有n维特征的基础上重新构造出来的k维特征。

4.Perceptron:
感知机是二类分类的线性分类模型，旨在求出将训练数据进行线性划分的分离超平面，因此导入基于误分类的损失函数，利用梯度下降法对
损失函数进行极小化，求得感知机模型。

5.Adaboost:
通过每次降低个体学习器的分类误差，加大效果好的个体学习器的重要性，得到最终的集成学习器。

6.Decision Tree:
决策树是一个预测模型，它代表的是对象属性与对象值之间的一种映射关系。树中每个节点表示某个对象，而每个分叉路径则代表某个可
能的属性值，而每个叶节点则对应从根节点到该叶节点所经历的路径所表示的对象的值。

7.EM：
算法步骤分为两步：Expection-Step 和 Maximization-Step。E-Step 主要通过观察数据和现有模型来估计参数，然后用这个估计的
参数值来计算似然函数的期望值；而 M-Step 是寻找似然函数最大化时对应的参数。由于算法会保证在每次迭代之后似然函数都会增加，
所以函数最终会收敛。

8.Hierarchical Clustering：
层次聚类算法，就是按照某种方法进行层次分类，直到满足某种条件为止。简单说它是将数据集中的每个样本初始化为一个簇，然后找到
距离最近的两个簇，将他们合并，不断重复这个过程，直达到到预设的聚类数目为止。

9.LinearRegression：
线性回归在假设特证满足线性关系，根据给定的训练数据训练一个模型，并用此模型进行预测。

10.Logistic：
Logistic Regression是一种用于解决二分类（0 or 1）问题的机器学习方法，用于估计某种事物的可能性。

11.SVM_linear and SVM_smo：
支持向量机（support vector machines, SVM）是一种二分类模型，它的基本模型是定义在特征空间上的间隔最大的线性分类器，间
隔最大使它有别于感知机；SVM还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，可形式化为一个
求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。SVM的的学习算法就是求解凸二次规划的最优化算法。

