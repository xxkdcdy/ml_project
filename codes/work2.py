import numpy as np


# 加载数据集
def load_dataset(filename):
    dataMat = []     # 特征数据
    labelMat = []    # 标签
    fr = open(filename)
    for line in fr.readlines():  # 解析文本文件中的数据
        lineArr = line.strip().split(',')
        dataMat.append([float(i) for i in lineArr[2:]])  # 数据矩阵
        labelMat.append(1.0 if lineArr[1] == 'M' else -1.0)  # 类标签
    return np.array(dataMat), np.array(labelMat)


# 支持向量机类
class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=100):
        self.C = C  # 正则化参数
        self.tol = tol  # 迭代停止条件
        self.max_iter = max_iter  # 最大迭代次数

    # 训练函数
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples, n_features = self.X_train.shape

        # 初始化参数
        self.alpha = np.zeros(n_samples)  # 初始化拉格朗日乘子
        self.b = 0  # 初始化截距
        self.errors = np.zeros(n_samples)  # 初始化误差

        # SMO算法
        iters = 0
        while iters < self.max_iter:
            num_changed_alphas = 0  # 记录更新的乘子个数
            for i in range(n_samples):
                Ei = self.decision_function(self.X_train[i]) - self.y_train[i]  # 计算误差
                # 检查KKT条件是否满足
                if (self.y_train[i]*Ei < -self.tol and self.alpha[i] < self.C) or \
                   (self.y_train[i]*Ei > self.tol and self.alpha[i] > 0):
                    j = self.select_second_alpha(i, n_samples)  # 选择第二个乘子
                    Ej = self.decision_function(self.X_train[j]) - self.y_train[j]  # 计算第二个样本的误差
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    L, H = self.compute_bounds(i, j, self.y_train)  # 计算乘子的取值范围
                    if L == H:
                        continue
                    eta = 2 * np.dot(self.X_train[i], self.X_train[j]) - np.dot(self.X_train[i], self.X_train[i]) - np.dot(self.X_train[j], self.X_train[j])  # 计算更新步长
                    if eta >= 0:
                        continue
                    self.alpha[j] -= self.y_train[j] * (Ei - Ej) / eta  # 更新第二个乘子
                    self.alpha[j] = max(self.alpha[j], L)  # 截取乘子的取值范围
                    self.alpha[j] = min(self.alpha[j], H)
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    self.alpha[i] += self.y_train[i]*self.y_train[j]*(alpha_j_old - self.alpha[j])  # 更新第一个乘子

                    # 更新截距
                    b1 = self.b - Ei - self.y_train[i] * (self.alpha[i] - alpha_i_old) * np.dot(self.X_train[i], self.X_train[i]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * np.dot(self.X_train[i], self.X_train[j])
                    b2 = self.b - Ej - self.y_train[i] * (self.alpha[i] - alpha_i_old) * np.dot(self.X_train[i], self.X_train[j]) - \
                         self.y_train[j] * (self.alpha[j] - alpha_j_old) * np.dot(self.X_train[j], self.X_train[j])
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    self.errors[i] = self.decision_function(self.X_train[i]) - self.y_train[i]  # 更新误差
                    self.errors[j] = self.decision_function(self.X_train[j]) - self.y_train[j]
                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                iters += 1
            else:
                iters = 0

    # 决策函数
    def decision_function(self, X):
        return np.sum(self.alpha * self.y_train * np.dot(X, self.X_train.T)) + self.b

    def select_second_alpha(self, i, n_samples):
        j = i
        while j == i:
            j = np.random.randint(n_samples)  # 随机选择第二个乘子
        return j

    def compute_bounds(self, i, j, y):
        if y[i] != y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])  # 计算乘子的取值范围
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])
        return L, H

    def predict(self, X):
        return np.sign(self.decision_function(X))  # 使用决策函数进行预测


# 测试
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # 生成数据集
    X, y = load_dataset('../datasets/wdbc.data')

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # 训练SVM模型
    svm = SVM()
    svm.fit(X_train, y_train)

    # 预测并评估模型
    acc = 0
    for i in range(len(y_test)):
        print('预测值: {}实际值: {}'.format(svm.predict(X_test[i]), y_test[i]))
        if svm.predict(X_test[i]) == y_test[i]:
            acc += 1
    print('Accuracy: {}'.format(acc / len(y_test)))
