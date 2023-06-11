import numpy as np
class adaboost_classification:

    def __init__(self, x1, y1, iteration):  # x,y,迭代次数
        self.x1 = x1
        self.y1 = y1
        self.iteration = iteration
        self.n = len(x1)  # 样本数
        self.w = np.ones(self.n)/self.n
        self.Gx = []  # 列表，存储分类器，第一列为符号，第二列为阈值,第三列为系数

    def predict(self, symbol, threshold): # 预测类别，输入符号和阈值，输出标签和误差

        prediction = np.ones(self.n)
        if symbol == '大于':
            prediction[self.x1 < threshold] = -1
        else:
            prediction[self.x1 > threshold] = -1

        error = np.sum(self.w[prediction != self.y1])
        return prediction,error

    def G_x(self, threshold_sequence): # 训练分类器，输出本轮迭代的最优符号和阈值

        symbol_sequence = ['大于','小于']
        best_error = 100000

        for symbol in symbol_sequence:

            for threshold in threshold_sequence:

                prediction, error = self.predict(symbol, threshold)
                if error < best_error:
                    best_error = error
                    best_symbol = symbol
                    best_threshold = threshold

        return best_symbol, best_threshold, best_error

    def fit(self):

        threshold_sequence = np.arange(min(self.x1)-0.5,max(self.x1)+1.5,1)
        for i in range(self.iteration):
            best_symbol, best_threshold, best_error = self.G_x(threshold_sequence)
            alpha = 0.5*np.log((1-best_error)/best_error)
            self.Gx.append([best_symbol, best_threshold, alpha])
            Gmx = self.predict(best_symbol, best_threshold)[0]
            self.w = self.w*np.exp(-alpha * self.y1 * Gmx)/np.sum(self.w*np.exp(-alpha * self.y1 * Gmx))  # 更新权重

    def final_predict(self):

        fx = np.zeros(self.n)
        for i in range(len(self.Gx)):
            fx = fx + self.Gx[i][2]*self.predict(self.Gx[i][0],self.Gx[i][1])[0]

        prediction = [1 if fx[i]>0 else -1 for i in range(self.n)]
        accuracy = np.sum(prediction == self.y1)/self.n
        print('实际值:',self.y1)
        print('预测值：',prediction)
        print('正确率：',accuracy)

print('----------------------------统计学习方法例8.1----------------------------')
x1 = np.arange(10)
y1 = np.array([1,1,1,-1,-1,-1,1,1,1,-1])
ada1 = adaboost_classification(x1, y1, iteration=3)
ada1.fit()
ada1.final_predict()
print(" ")

# ===================================================================================
# 统计学习方法例8.2
# ===================================================================================
class adaboost_regression:

    def __init__(self, x2, y2, iteration):
        self.x2 = x2
        self.y2 = y2
        self.iteration = iteration
        self.n = len(x2)  # 样本数
        self.Tx = []  # 列表，存储分类器，第一列为阈值，第二列为c1,第三列为c2
        self.rx = y2  # 残差

    def T_x(self,threshold_sequence):

        error = 1000000
        for s in threshold_sequence:

            R1 = self.x2 <= s
            R2 = self.x2 > s
            c1 = np.mean(self.rx[R1])
            c2 = np.mean(self.rx[R2])
            ms = np.dot(self.rx[R1]-c1, self.rx[R1]-c1)+np.dot(self.rx[R2]-c2, self.rx[R2]-c2)

            if ms < error:
                best_threshold = s
                best_c1 = c1
                best_c2 = c2
                error = ms

        return best_threshold, best_c1,best_c2

    def predict(self):

        fx = np.zeros(self.n)
        for i in range(len(self.Tx)):
            fx[self.x2 < self.Tx[i][0]] = fx[self.x2 < self.Tx[i][0]] + self.Tx[i][1]
            fx[self.x2 >= self.Tx[i][0]] = fx[self.x2 >= self.Tx[i][0]] + self.Tx[i][2]

        prediction = fx
        if len(self.Tx) == self.iteration:
            error = np.dot(self.y2-prediction, self.y2-prediction)
            print('实际值：', self.y2)
            print('预测值：',prediction)
            print('误差：', error)
        return prediction

    def fit(self):

        threshold_sequence = np.arange(min(self.x2) + 0.5, max(self.x2), 1)
        for i in range(self.iteration):
            best_threshold, best_c1,best_c2 = self.T_x(threshold_sequence)
            self.Tx.append([best_threshold, best_c1,best_c2])
            self.rx = self.y2 - self.predict()

print('----------------------------统计学习方法例8.2----------------------------')
x2 = np.arange(10)+1
y2 = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
ada2 = adaboost_regression(x2, y2, iteration=100)
ada2.fit()
print(" ")

# ===================================================================================
# 统计学习方法习题8.1
# ===================================================================================
class adaboost_multidimensional:

    def __init__(self, x3, y3, iteration):
        self.x3 = x3
        self.y3 = y3
        self.iteration = iteration
        self.m = x3.shape[0]  # 样本数
        self.n = x3.shape[1]  # 维度
        self.w = np.ones(self.m)/self.m
        self.Gx = []  # 列表，存储分类器，第一列为符号，第二列为维度，第三列为阈值，第四列为系数

    def predict(self, symbol, dimension, threshold): # 预测类别，输入符号和阈值，输出标签和误差

        prediction = np.ones(self.m)
        if symbol == '大于':
            prediction[self.x3[:,dimension] < threshold] = -1
        else:
            prediction[self.x3[:,dimension] > threshold] = -1

        error = np.sum(self.w[prediction != self.y3])
        return prediction,error

    def G_x(self): # 训练分类器，输出本轮迭代的最优符号,维度、阈值和误差

        symbol_sequence = ['大于','小于']
        best_error = 100000

        for symbol in symbol_sequence:

            for dimension in range(self.n):

                threshold_sequence = np.arange(min(self.x3[:,dimension])-0.5,max(self.x3[:,dimension])+0.5,0.2)
                for threshold in threshold_sequence:

                    prediction, error = self.predict(symbol, dimension, threshold)
                    if error < best_error:
                        best_error = error
                        best_symbol = symbol
                        best_dimension = dimension
                        best_threshold = threshold

        return best_symbol, best_dimension, best_threshold, best_error


    def fit(self):

        for i in range(self.iteration):
            best_symbol, best_dimension, best_threshold, best_error = self.G_x()
            alpha = 0.5*np.log((1-best_error)/best_error)
            self.Gx.append([best_symbol,best_dimension, best_threshold, alpha])
            Gmx = self.predict(best_symbol,best_dimension, best_threshold)[0]
            self.w = self.w*np.exp(-alpha * self.y3 * Gmx)/np.sum(self.w*np.exp(-alpha * self.y3 * Gmx))  # 更新权重

    def final_predict(self):

        fx = np.zeros(self.m)
        for i in range(len(self.Gx)):
            fx = fx + self.Gx[i][3]*self.predict(self.Gx[i][0],self.Gx[i][1],self.Gx[i][2])[0]

        prediction = [1 if fx[i]>0 else -1 for i in range(self.m)]
        accuracy = np.sum(prediction == self.y3)/self.m
        print('实际值:',self.y3)
        print('预测值：',prediction)
        print('正确率：',accuracy)

print('----------------------------统计学习方法习题8.1----------------------------')
x3 = np.array([[0,1,3],[0,3,1],[1,2,2],[1,1,3],[1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]])
y3 = np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1])
ada1 = adaboost_multidimensional(x3, y3, iteration=10)
ada1.fit()
ada1.final_predict()
