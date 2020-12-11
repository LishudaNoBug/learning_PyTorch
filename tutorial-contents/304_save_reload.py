
import torch
import matplotlib.pyplot as plt

"""
    保存模型和提取模型
    两种保存方式：
        1.torch.save(net1, 'net.pkl')                       # 保存训练好的整个神经网络。保存net1网络并取名net.pkl
        2.torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存训练好的网络的参数（例如各节点的参数信息）
    提取时就对应两种提取方式：
        1.net2 = torch.load('net.pkl')                      #重新加载整个网络
        2.                                                  #需要创建一个和当时一样的网络，然后加载保存好的参数
            net3 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
            )
            
            net3.load_state_dict(torch.load('net_params.pkl'))
            
"""

# 初始化输入数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

"""
    先自己创建网络，训练好后保存
    既有保存整个网络的方法，也有只保存参数的方法
"""
def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    # 上面都和之前一样，就在此处保存网络的。（两种方式）
    torch.save(net1, 'net.pkl')                         # 保存训练好的整个神经网络。保存net1网络并取名net.pkl
    torch.save(net1.state_dict(), 'net_params.pkl')     # 只保存训练好的网络的参数

"""
    加载save保存的整个网络
"""
def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

"""
    只加载save保存的参数
"""
def restore_params():
    # restore only the parameters in net1 to net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    # copy net1's parameters into net3
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()

# save net1
save()

# restore entire net (may slow)
restore_net()

# restore only the net parameters
restore_params()
