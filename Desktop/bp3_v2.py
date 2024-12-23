import torch
import torch.nn as nn
import torch.optim as optim


def sigmoid(z):
    return 1.0 / (1 + torch.exp(-z))


def t1():
    # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
    v = nn.Parameter(torch.tensor([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35]
    ]))
    b11 = torch.tensor([0.15], requires_grad=True)
    b12 = torch.tensor([0.30])
    b1 = b11 + b12
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
    w = nn.Parameter(torch.tensor([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ]))
    b2 = torch.tensor([0.65])

    # 构建优化器
    opt = optim.SGD(params=[v, b11, w], lr=0.1)

    # 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
    x = torch.tensor([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = torch.tensor([
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99]  # 人为任意给定的
    ])

    # 第一个隐藏的操作输出
    net_h = torch.matmul(x, v) + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
    out_h = sigmoid(net_h)
    # 输出层的操作输出
    net_o = torch.matmul(out_h, w) + b2  # [N,2] N表示样本数目，3表示每个样本有2个特征/2个输出
    out_o = sigmoid(net_o)
    loss = 0.5 * torch.sum(torch.pow((out_o - d), 2))
    # print(loss)
    # print(net_h)
    # print(out_h)
    # print(net_o)
    # print(out_o)
    # print(x)
    # print(v)
    # print(b1)
    # print(w)
    # print(b2)
    # print("=" * 50)

    # BP过程
    opt.zero_grad()  # 将所有的训练参数对应的初始梯度值全部重置为0
    loss.backward()  # 反向传播，求解loss关于每个参数的梯度值 --> backward完成后，只有参数(requires_grad为True的tensor会保留grad梯度值，其它中间节点的梯度全部会被删除)
    opt.step()  # 基于求解的梯度值，进行参数更新

    print(v)
    print(b11)
    print(w)


def t2():
    # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
    v = nn.Parameter(torch.tensor([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35]
    ]))
    b11 = torch.tensor([0.15], requires_grad=True)
    b12 = torch.tensor([0.30])
    b1 = b11 + b12
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
    w = nn.Parameter(torch.tensor([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ]))
    b2 = torch.tensor([0.65])

    # 构建优化器
    opt = optim.SGD(params=[v, b11, w], lr=0.1)

    # 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
    x = torch.tensor([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = torch.tensor([
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99]  # 人为任意给定的
    ])

    for i in range(100):
        # 第一个隐藏的操作输出
        net_h = x @ v + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
        out_h = sigmoid(net_h)
        # 输出层的操作输出
        net_o = torch.matmul(out_h, w) + b2  # [N,2] N表示样本数目，3表示每个样本有2个特征/2个输出
        out_o = sigmoid(net_o)
        loss = 0.5 * torch.sum(torch.pow((out_o - d), 2))

        # BP过程
        opt.zero_grad()  # 将所有的训练参数对应的初始梯度值全部重置为0
        loss.backward()  # 反向传播，求解loss关于每个参数的梯度值 --> backward完成后，只有参数(requires_grad为True的tensor会保留grad梯度值，其它中间节点的梯度全部会被删除)
        opt.step()  # 基于求解的梯度值，进行参数更新

        if i % 10 == 0:
            print(i, loss)

    print(v)
    print(b11)
    print(w)
    print(loss)


def t3():
    # 第一层：隐层具有3个神经元，输入是2个，所以该层的权重系数v形状为:[2,3]
    v = nn.Parameter(torch.tensor([
        [0.1, 0.2, 0.3],
        [0.15, 0.25, 0.35]
    ]))
    b11 = torch.tensor([0.15], requires_grad=True)
    b12 = torch.tensor([0.30])
    b1 = b11 + b12
    # 第二层：输出层具有2个神经元，输入是3个，所以该层的权重系数w的形状为:[3,2]
    w = nn.Parameter(torch.tensor([
        [0.4, 0.45],
        [0.5, 0.55],
        [0.6, 0.65]
    ]))
    b2 = torch.tensor([0.65])

    # 构建优化器
    opt = optim.SGD(params=[v, b11, w], lr=0.1)

    # 当前输入1个样本，每个样本2个特征属性，就相当于输入层的神经元是2个
    x = torch.tensor([
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0],
        [5.0, 10.0]
    ])
    # 实际值
    d = torch.tensor([
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99],  # 人为任意给定的
        [0.01, 0.99]  # 人为任意给定的
    ])

    opt.zero_grad()  # 将所有的训练参数对应的初始梯度值全部重置为0
    for i in range(100):
        # 第一个隐藏的操作输出
        net_h = x @ v + b1  # [N,3] N表示样本数量，3表示每个样本有3个特征
        out_h = sigmoid(net_h)
        # 输出层的操作输出
        net_o = torch.matmul(out_h, w) + b2  # [N,2] N表示样本数目，3表示每个样本有2个特征/2个输出
        out_o = sigmoid(net_o)
        loss = 0.5 * torch.sum(torch.pow((out_o - d), 2))

        # BP过程
        loss.backward()  # 反向传播，求解loss关于每个参数的梯度值 --> backward完成后，只有参数(requires_grad为True的tensor会保留grad梯度值，其它中间节点的梯度全部会被删除)

        if (i % 10 == 0) or (i == 99):
            print(i, loss)
            # 每10次进行一次参数更新
            opt.step()  # 基于求解的梯度值，进行参数更新
            opt.zero_grad()  # 梯度重置为0

    print(v)
    print(b11)
    print(b12)
    print(w)


if __name__ == '__main__':
    t3()
