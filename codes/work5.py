import torch
import torchvision
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import concurrent.futures
import time
from loguru import logger

# https://blog.csdn.net/jeazim/article/details/134995502
# https://ymzhang-cs.github.io/posts/use-of-dataloader-and-dataset/


client_num = 10
# clear the log before logging
open('runtime.log', 'w').close()
# logger.add('logs/{}.log'.format(time.time()))
logger.add('runtime.log')
logger.info(torch.__version__)


class module(nn.Module):
    def __init__(self):
        super().__init__()  # 继承父类的函数
        self.conv1 = nn.Conv2d(1, 10, 5)  # 卷积 输入通道1，输出通道10，卷积核5（5*5）
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10 20 3*3
        self.fc1 = nn.Linear(20 * 10 * 10, 500)  # 全连接层
        self.fc2 = nn.Linear(500, 10)  # 0到9十个数字 输出10

    def forward(self, x):  # 定义了forward函数，backward函数就会被自动实现(利用Autograd)
        input_size = x.size(0)  # batch_size *1 *28 *28
        x = self.conv1(x)  # 输入 batch_size *1 *28 *28，输出 batch_size *10 *24 *24（28卷积5：28-5+1）
        x = F.relu(x)  # 激活函数 使其变为非线性函数
        x = F.max_pool2d(x, 2, 2)  # 池化层 保持shape不变 输出 batch_size *10 *12 *12（24/2）

        x = self.conv2(x)  # 输出： batch_size *20 *10 *10（12-3+1）
        x = F.relu(x)  # 激活函数

        x = x.view(input_size, -1)  # 拉伸 -1(自动计算长度):20*10*10 = 2000

        x = self.fc1(x)  # 输入：2000 输出：500
        x = F.relu(x)  # 激活函数
        x = self.fc2(x)  # 输入：500 输出：10

        output = F.log_softmax(x, dim=1)  # 计算分类后，每个数字的概率值
        return output


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def mnist_iid(dataset, num_client):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_client:
    :return: dict of image index
    """
    # 每个客户端持有的数据量
    num_items = int(len(dataset) / num_client)
    # 每个客户端拥有的数据ID
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_client):
        # 每个客户端从数据集中的ID随机抽取
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        # 把抽过的数据ID从池子中抽走
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_client):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_client:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # 把60000个训练数据分成200份，每份包含300条数据
    num_shards, num_imgs = 200, 300
    # 每一份的序号
    idx_shard = [i for i in range(num_shards)]
    # 定义每个客户端拿到的实际序号
    dict_users = {i: np.array([]) for i in range(num_client)}
    # 所有的实际序号
    idxs = np.arange(num_shards * num_imgs)
    # 所有的数据集标签
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_client):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


# 定义一个FedAvg聚合函数
def fedavg_aggregate(w_list):
    # 创建一个深拷贝的模型参数对象
    w_avg = copy.deepcopy(w_list[0])

    # 对模型参数的每个键进行循环迭代
    for k in w_avg.keys():
        # 迭代每个模型参数（除了第一个）
        for i in range(1, len(w_list)):
            # 将每个模型参数的值加到联邦平均模型参数上
            w_avg[k] += w_list[i][k]

        # 计算平均值，这里假设所有客户端的数据量相同，将累加的值除以模型数量
        # 这行代码的作用是将模型参数 w_avg[k] 的值与模型数量 len(w) 相除，以获得联邦平均模型参数的值。
        # 由于 w_avg 是一个字典，w_avg[k] 是一个 PyTorch 张量对象。
        # 因此，这行代码的目的是对模型参数进行逐元素的除法运算，并返回一个新的具有相同形状的张量，其中每个元素等于相应位置上 w_avg[k] 的元素除以 len(w)
        w_avg[k] = torch.div(w_avg[k], len(w_list))

    # 返回计算得到的联邦平均模型参数
    return w_avg


def train_client(client):
    w, loss = client.train()  # Train the client model
    return (w, client.data_size)  # Return the model weights and data size


class Client:
    def __init__(self, cid, idxs, device, train_set, test_set):
        self.cid = cid
        self.device = device
        self.model = load_model(model_name='ResNet50')
        self.train_set = train_set
        self.test_set = test_set
        self.data_size = len(idxs)
        self.local_epoch = 5
        # 保存一个交叉熵损失函数的实例，用于计算训练过程中的损失
        self.loss_func = nn.CrossEntropyLoss()
        # 创建一个数据加载器DataLoader，加载一个子数据集DatasetSplit，其中子数据集由参数dataset和idxs指定，设置批量大小为self.args.local_bs，并进行随机洗牌
        self.ldr_train = DataLoader(DatasetSplit(self.train_set, idxs), batch_size=10, shuffle=True)

    # 模型本地训练
    def train(self):
        # 将模型设置为训练模式
        self.model.train()
        # train and update
        # 创建一个torch.optim.SGD的优化器，使用parameters()作为优化器的参数，设置学习率lr和动量momentum
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

        # 用于保存每个训练周期的损失
        epoch_loss = []
        for iter in range(self.local_epoch):
            # 用于保存每个批次的损失
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                # 清零模型参数的梯度
                self.model.zero_grad()

                # 通过模型进行前向传播，获取预测的对数概率
                log_probs = self.model(images)

                # 使用损失函数计算损失
                loss = self.loss_func(log_probs, labels)

                # 对损失进行反向传播和参数更新
                loss.backward()
                optimizer.step()

                # 批次索引能被10整除，打印当前训练进度和损失

                # 计算每个训练周期的平均损失，并将其添加到epoch_loss中
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logger.info('Client {} Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                self.cid, iter + 1, (batch_idx + 1) * len(images), len(self.ldr_train.dataset),
                      100. * (batch_idx + 1) / len(self.ldr_train), loss.item()))
        # 返回模型的状态字典和所有训练周期的平均损失
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # 模型测试
    def test(self, test_model):
        test_model.eval()
        # testing
        # 计算测试损失和正确分类的样本数
        test_loss = 0
        correct = 0
        data_loader = DataLoader(self.test_set, batch_size=10)
        l = len(data_loader)
        # 对数据加载器进行迭代，每次迭代获取一个批量的数据和对应的目标标签
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            # 调用net_g模型对数据进行前向传播
            log_probs = test_model(data)
            # sum up batch loss
            # 使用交叉熵损失函数F.cross_entropy计算损失并累加到test_loss中
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            # 利用预测的对数概率计算预测的类别，并与目标标签进行比较，统计正确分类的样本数
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        # 计算平均测试损失和准确率
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

        # 打印详细的测试结果
        logger.info('\nClient {} Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            self.cid, test_loss, correct, len(data_loader.dataset), accuracy))

        return accuracy, test_loss


def load_dataset():
    pass


def load_model(model_name='ResNet18'):
    pre_trained_model = None
    if model_name == 'ResNet18':
        pre_trained_model = torchvision.models.resnet18(pretrained=True)
        pre_trained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = pre_trained_model.fc.in_features
        pre_trained_model.fc = nn.Linear(num_features, 10)   # 换成MNIST的10类输出

    elif model_name == 'ResNet50':
        pre_trained_model = torchvision.models.resnet50(pretrained=True)
        pre_trained_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = pre_trained_model.fc.in_features
        pre_trained_model.fc = nn.Linear(num_features, 10)  # 换成MNIST的10类输出

    pre_trained_model = pre_trained_model.to(device)
    return pre_trained_model


# 定义训练环境
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 获取预训练模型
# model = torchvision.models.resnet18(pretrained=True).to(device)
# model = load_model(model_name='ResNet18')
model = load_model(model_name='ResNet50')


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化图像数据
])

# 获取数据集
train_data = datasets.MNIST(root='../datasets/data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../datasets/data', train=False, download=True, transform=transform)


# 单机训练
def learning_single():
    # def __init__(self, idxs, device, model, train_set, test_set):
    idxs = [i for i in range(len(train_data))]
    client = Client(0, idxs, device, train_data, test_data)
    # 直接调用训练函数，单机训练
    client.train()
    # 测试
    client.test(client.model)


# 中心化联邦学习
def centralize_federated_learning():
    # 初始化客户端
    dict_users = mnist_iid(train_data, 10)
    client_list = [Client(i, dict_users[i], device, train_data, test_data) for i in range(client_num)]

    # 初始化一个全局模型
    global_model = load_model(model_name='ResNet50')

    # 这里把本地轮次从5，变成1
    for client in client_list:
        client.local_epoch = 1

    # 进行5个全局轮次，总的训练量保持一致
    for episode in range(5):
        w_list = []
        # 客户端接收全局模型
        for client in client_list:
            client.model.load_state_dict(global_model.state_dict())  # Load the global model
        # 客户端本地训练
        # Using ThreadPoolExecutor for I/O-bound operations or ProcessPoolExecutor for CPU-bound operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(train_client, client) for client in client_list]

            for future in concurrent.futures.as_completed(futures):
                w, data_size = future.result()
                w_list.append(w)
        # 本地模型上传进行FedAvg聚合
        aggregated_weight = fedavg_aggregate(w_list)
        global_model.load_state_dict(aggregated_weight)
    # 测试
    client.test(global_model)


# 去中心化联邦学习
def decentralize_federated_learning():
    # 初始化客户端
    dict_users = mnist_iid(train_data, 10)
    client_list = [Client(i, dict_users[i], device, train_data, test_data) for i in range(client_num)]

    # 这里把本地轮次从5，变成1
    for client in client_list:
        client.local_epoch = 1

    # 进行5个全局轮次，总的训练量保持一致
    for episode in range(5):
        # 客户端本地训练
        # Using ThreadPoolExecutor for I/O-bound operations or ProcessPoolExecutor for CPU-bound operations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the executor
            futures = [executor.submit(train_client, client) for client in client_list]

        # 去中心化执行聚合过程
        for client in client_list:
            left_client = client_list[(client.cid - 1 + client_num) % client_num]
            right_client = client_list[(client.cid + 1) % client_num]
            w_list = [left_client.model.state_dict(), right_client.model.state_dict(), client.model.state_dict()]

            # 客户端进行FedAvg聚合
            aggregated_weight = fedavg_aggregate(w_list)
            client.model.load_state_dict(aggregated_weight)
    # 测试
    acc_avg = 0
    for client in client_list:
        accuracy, test_loss = client.test(client.model)
        acc_avg += test_loss
    logger.info(f'avg_acc: {acc_avg}')


if __name__ == '__main__':
    # 记录 learning_single 的执行时间
    # start_time = time.time()
    # learning_single()
    # end_time = time.time()
    # learning_single_duration = end_time - start_time
    # logger.info(f"learning_single() 执行时间: {learning_single_duration:.4f} 秒")

    # 记录 centralize_federated_learning 的执行时间
    start_time = time.time()
    centralize_federated_learning()
    end_time = time.time()
    centralize_federated_learning_duration = end_time - start_time
    logger.info(f"centralize_federated_learning() 执行时间: {centralize_federated_learning_duration:.4f} 秒")

    # 记录 decentralize_federated_learning 的执行时间
    start_time = time.time()
    decentralize_federated_learning()
    end_time = time.time()
    decentralize_federated_learning_duration = end_time - start_time
    logger.info(f"decentralize_federated_learning() 执行时间: {decentralize_federated_learning_duration:.4f} 秒")
