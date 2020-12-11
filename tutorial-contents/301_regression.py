import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

"""
    训练回归模型（初级）
        回归理解为: 对于任意输入有y对应，且y是连续的，形成一条连续的函数图像？？
"""

x = torch.unsqueeze(torch.linspace(-1, 1, 100),1)   # torch.linspace(-1, 1, 100)是-1~1之间取100个数。 unsqueeze是将一维的数处理成二维。因为torch只能处理二维的数，所以这里就变成了（100，1）的数。
y = x.pow(2) + 0.2*torch.rand(x.size())             # 制造假的y数据： y=x^2+0.2噪声随机数

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):    #n_feature:输入的特征数；n_hidden隐藏层神经元数；n_output输出的个数
        super(Net, self).__init__()     # 官网固定写法
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):               # 重载torch.nn.Module的forward
        x = self.hidden(x)              # 隐藏层y=wx+b函数
        x = F.tanh(x)                   # 激活函数
        x = self.predict(x)             # 调用init的predict函数
        return x


net = Net(n_feature=1, n_hidden=10, n_output=1)     # 创建Net对象。即一个输入，隐藏层10个1神经元，1个输出
print(net)  # 打印net对象的hidden和predict属性

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)        # optim是优化器，用来初始所有w权重参数。lr是学习率，一般设置小于1。
loss_func = torch.nn.MSELoss()           # MSELoss损失函数，这里用均方误差，但吴恩达说这找不到全局最优解

plt.ion()   # plt ion 开启交互模式，当plt.plot打印图片后程序继续执行。如果不开，则plot打印后程序不会继续执行。

for t in range(1000):        #梯度下降200次，
    prediction = net(x)     # net（x）会调用Net的forward方法

    loss = loss_func(prediction, y)     # 损失函数，must be (1. nn output, 2. target)

    optimizer.zero_grad()   # 清空缓存的梯度
    loss.backward()         # 反向传播，先计算各个节点的梯度
    optimizer.step()        # 然后应用梯度（这里学习率设置的是0.2）

    # plot and show learning process
    if t % 20 == 0:      #每训练5次打印一下
        plt.cla() # matplotlib 维护的 figure 有数量上限,在某些情况下，不清理 figure 将有可能造成在第一幅中 plot 的线再次出现在第二幅图中
        plt.scatter(x.data.numpy(), y.data.numpy(),alpha=0.2)                     # scatter散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=1)   # plot线；'r-'指red（r）色直线；lw lineWidth；
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})  # 前两位0.5,0表示输出信息的坐标(坐标原点0.0)；Loss=%.4f 保留小数点后四位；字体尺寸20，颜色红色
        plt.pause(0.1)

plt.ioff()
plt.show()

