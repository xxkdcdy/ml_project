import random
import numpy as np


# 获取C的概率
def check_c(s):
    return 0.5 if s[0] else 1 - 0.5


# 获取U的概率
def check_u(s):
    if s[0]:
        return 0.95 if s[1] else 1 - 0.95
    else:
        return 0.01 if s[1] else 1 - 0.01


# 获取W的概率
def check_w(s):
    if s[1]:
        return 0.90 if s[2] else 1 - 0.90
    else:
        return 0.05 if s[2] else 1 - 0.05


# 获取B的概率
def check_b(s):
    if s[1]:
        return 0.30 if s[3] else 1 - 0.30
    else:
        return 0.01 if s[3] else 1 - 0.01


# 获取D的概率
def check_d(s):
    if s[2]:
        if s[3]:
            return 0.335 if s[4] else 1 - 0.335
        else:
            return 0.30 if s[4] else 1 - 0.30
    else:
        if s[3]:
            return 0.05 if s[4] else 1 - 0.05
        else:
            return 0.0 if s[4] else 1 - 0.0


# 获取当前的采样概率
def get_sample_rate(s):
    return check_c(s) * check_u(s) * check_w(s) * check_b(s) * check_d(s)

# 目标是求P(D|U=True, W=True)


n = 10000   # 迭代步数
m = 8000    # 收敛步数

samples = []    # 样本集合

# 初始化，给出初始样本（随机生成）
# 当前状态[C, U, W, B, D], U=True, W=True, 其余随机生成
state = [bool(random.getrandbits(1)), True, True, bool(random.getrandbits(1)), bool(random.getrandbits(1))]

# 循环n次
for i in range(n):
    # 对C进行采样
    p1 = get_sample_rate([state[0], state[1], state[2], state[3], state[4]])
    p2 = get_sample_rate([not state[0], state[1], state[2], state[3], state[4]])
    p = p1 / (p1 + p2)
    # 检查是否接受改变
    r = np.random.uniform()
    # 如果不接受改变就变回去
    if r > p:
        state[0] = not state[0]

    # 对B进行采样
    p1 = get_sample_rate([state[0], state[1], state[2], state[3], state[4]])
    p2 = get_sample_rate([state[0], state[1], state[2], not state[3], state[4]])
    p = p1 / (p1 + p2)
    # 检查是否接受改变
    r = np.random.uniform()
    # 如果不接受改变就变回去
    if r > p:
        state[3] = not state[3]

    # 对D进行采样
    p1 = get_sample_rate([state[0], state[1], state[2], state[3], state[4]])
    p2 = get_sample_rate([state[0], state[1], state[2], state[3], not state[4]])
    p = p1 / (p1 + p2)
    # 检查是否接受改变
    r = np.random.uniform()
    # 如果不接受改变就变回去
    if r > p:
        state[4] = not state[4]

    # 保存本轮结果到样本集合中
    samples.append([state[0], state[1], state[2], state[3], state[4]])

# 对样本集合进行截取，只要收敛步数后的样本
samples = samples[m:]

# 计算结果
t = 0
f = 0
for i in samples:
    if i[4]:
        t += 1
    else:
        f += 1

print('20235227087陈东宇')
print('P(D|U=True, W=True): <{}, {}>'.format(t / (t + f), f / (t + f)))
