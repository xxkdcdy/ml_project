import pandas as pd
import math


# 数据处理
def load_dataset():
    with open('../datasets/corpus.txt', 'r', encoding='utf-8') as file:
        i = 0    # 行号，跳过第一行
        data = []    # 最终处理好的数据
        examples = []
        for line in file:
            # [你]旁边多了空格
            line = line.replace(' [你] ', '[你]')
            # 去除多余的空格
            while '  ' in line:
                line = line.replace('  ', ' ')
            if i != 0:
                batch = []
                example = ''
                # 去除行号
                words = line.split('\t')[2].split(' ')[1: -1]
                for w in words:
                    # 去除没有标注词性的内容
                    if '/' not in w:
                        continue
                    s = w.split('/')
                    # 按照(单词，词性)处理
                    batch.append((s[0], s[1]))
                    example += s[0] + ' '
                data.append(batch)
                examples.append(example[:-1])
            i += 1
        return data, examples


# 训练(建模)
def train(sentences):
    global A, B, pi, tags  # 声明全局变量
    # 统计 词性到词性转移矩阵A 词性到词转移矩阵B 初始向量pi
    # 先初始化
    A = {tag: {tag: 0 for tag in tags} for tag in tags}
    B = {tag: {word: 0 for word in words} for tag in tags}
    pi = {tag: 0 for tag in tags}
    # 统计A，B
    for words_with_tag in sentences:
        head_word, head_tag = words_with_tag[0]
        pi[head_tag] += 1
        B[head_tag][head_word] += 1
        # 计算词性转移和词转移
        for i in range(1, len(words_with_tag)):
            A[words_with_tag[i - 1][1]][words_with_tag[i][1]] += 1
            B[words_with_tag[i][1]][words_with_tag[i][0]] += 1
    # 拉普拉斯平滑处理并转换成概率
    sum_pi_tag = sum(pi.values())
    for tag in tags:
        pi[tag] = (pi[tag] + 1) / (sum_pi_tag + len(tags))
        sum_A_tag = sum(A[tag].values())
        sum_B_tag = sum(B[tag].values())
        for next_tag in tags:
            A[tag][next_tag] = (A[tag][next_tag] + 1) / (sum_A_tag + len(tags))
        for word in words:
            B[tag][word] = (B[tag][word] + 1) / (sum_B_tag + len(words))


# 测试(求解)
def test(sentence):
    words = sentence.split()
    sen_length = len(words)
    T1 = [{tag: float('-inf') for tag in tags} for _ in range(sen_length)]
    T2 = [{tag: None for tag in tags} for _ in range(sen_length)]
    # 先进行第一步
    for tag in tags:
        T1[0][tag] = math.log(pi[tag]) + math.log(B[tag][words[0]])
    # 继续后续解码
    for i in range(1, sen_length):
        for tag in tags:
            for pre_tag in tags:
                current_prob = T1[i-1][pre_tag] + math.log(A[pre_tag][tag]) + math.log(B[tag][words[i]])
                if current_prob > T1[i][tag]:
                    T1[i][tag] = current_prob
                    T2[i][tag] = pre_tag
    # 获取最后一步的解码结果
    last_step_result = [(tag, prob) for tag, prob in T1[sen_length-1].items()]
    last_step_result.sort(key=lambda x: -1*x[1])
    last_step_tag = last_step_result[0][0]
    # 向前解码
    step = sen_length - 1
    result = [last_step_tag]
    while step > 0:
        last_step_tag = T2[step][last_step_tag]
        result.append(last_step_tag)
        step -= 1
    result.reverse()
    return list(zip(words, result))


dataset, unlabeled = load_dataset()

A = {}
B = {}
pi = {}
# 统计words和tags
words = set()
tags = set()
# 把词和词性分开
for words_with_tag in dataset:
    for word_with_tag in words_with_tag:
        word, tag = word_with_tag
        words.add(word)
        tags.add(tag)
words = list(words)
tags = list(tags)

acc = 0    # 正确词数
si = 0     # 总词数

# 将数据切分成数据集和测试集
data_size = len(dataset)
test_cnt = int(0.1 * data_size)
train_set = dataset[test_cnt:]
test_set = unlabeled[:test_cnt]

train(train_set)

for i in range(len(test_set)):
    res = test(test_set[i])
    print('sentence {}: {}'.format(i + 1, res))
    # 对每个词的正确性进行统计
    for j in range(len(res)):
        if res[j] == dataset[i][j]:
            acc += 1
        si += 1

print('20235227087陈东宇')
print('Accuracy: {}'.format(acc / si))
