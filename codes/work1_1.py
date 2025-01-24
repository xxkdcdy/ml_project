import pandas as pd
import numpy as np


# sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 交叉熵损失函数
def cross_entropy_loss(y_true, y_pred):
    # y_true是真实标签，y_pred是预测值
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# 定义逻辑回归模型
class LogisticRegression:
    def __init__(self, num_features):
        # 初始化权重参数
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def fit(self, X, y, learning_rate=0.01, num_iterations=100000):
        # 训练模型
        for i in range(num_iterations):
            # 计算模型预测值
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

            # 计算交叉熵损失函数
            loss = np.mean(cross_entropy_loss(y, y_pred))

            # 计算梯度
            dW = np.dot(X.T, (y_pred - y)) / len(y)
            db = np.mean(y_pred - y)

            # 更新权重参数
            self.weights -= learning_rate * dW
            self.bias -= learning_rate * db

            # 打印损失值
            # if i % 100 == 0:
            #     print("Iteration %d, loss: %f" % (i, loss))

    def predict(self, X):
        # 预测样本标签
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.round(y_pred)

    # 性能评估
    def judge(self, x, y):
        y_pred = self.predict(np.array(x))
        tp = 0    # 真正例
        tn = 0    # 真反例
        fp = 0    # 伪正例
        fn = 0    # 伪反例
        for i in range(len(y)):
            print(y[i], y_pred[i][0])
            # 数据集前8个是正例，后9个是反例
            if i < 8:
                if y[i] == y_pred[i][0]:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i] == y_pred[i][0]:
                    tn += 1
                else:
                    fn += 1
        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        print('accuracy: {}'.format(accuracy))     # 准确率
        print('precision: {}'.format(precision))   # 精确率
        print('recall: {}'.format(recall))         # 召回率
        print('f1: {}'.format(f1))                 # f1分数


def work1_1_cdy():
    # 加载参数
    data = pd.read_csv(r'datasets/work1.csv', sep='\t', usecols=[1, 2, 3])
    y_data = []
    x_data = []
    for i in range(data.shape[0]):
        x_data.append([data['density'][i], data['Sugar content'][i]])
        y_data.append([data['classification'][i]])
    lr = LogisticRegression(len(x_data[0]))
    lr.fit(np.array(x_data), np.array(y_data))
    # print(lr.predict(np.array(x_data)))
    lr.judge(x_data, y_data)
