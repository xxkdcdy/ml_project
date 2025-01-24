# https://zhuanlan.zhihu.com/p/413919906
import numpy as np


class Perceptron:
    # N: 特征维度
    # alpha: 学习率
    # W: 权重
    def __init__(self, N, alpha=0.1):
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    # 按值的正负分类
    def step(self, x):
        return 1 if x > 0 else 0

    # 训练过程
    def fit(self, X, y, epochs=10):
        # 插入一维，将偏差视为直接在权重矩阵内的可训练参数。
        X = np.c_[X, np.ones((X.shape[0]))]
        for epoch in np.arange(0, epochs):
            # 取输入特征之间的点积X和权重矩阵W，然后将输出通过step函数来获得感知器的预测
            for (x, target) in zip(X, y):
                p = self.step(np.dot(x, self.W))
                # 当预测结果不一致的时候，执行权重更新
                if p != target:
                    error = p - target
                    self.W += -self.alpha * error * x

    # 预测过程
    def predict(self, X, addBias=True):
        # 返回高维数组，用于计算
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))


# AND操作
def AND():
    # 训练数据定义
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [0], [0], [1]])
    print("[INFO] training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)
    print("[INFO] testing perceptron...")
    for (x, target) in zip(X, y):
        pred = p.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))


def OR():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    print("[INFO] training perceptron...")
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)
    print("[INFO] testing perceptron...")
    for (x, target) in zip(X, y):
        pred = p.predict(x)
        print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))


def work1_2_cdy():
    print("[INFO] AND")
    AND()
    print("[INFO] OR")
    OR()
