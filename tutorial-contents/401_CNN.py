import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision              #torchvision是torch安装时自带的一些数据集。MNIST是其中手写数字的数据集。
import matplotlib.pyplot as plt

# 超级参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True   #为True会下载torchvision里的训练集，为False就不会下载。但为False我会报错不知道为啥。

# 定义训练集对象
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     #？？？？mnist里面既包括训练集又包括测试集的意思吗？
    transform=torchvision.transforms.ToTensor(),    # 转换原始图片？？？？？？？
    download=DOWNLOAD_MNIST,
)

# 加载训练集
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

"""
# plot打印训练集第一张图片
print(train_data.train_data.size())                 # (60000, 28, 28)
print(train_data.train_labels.size())               # (60000)
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()
"""

# 定义测试集（验证集）
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, 1).type(torch.FloatTensor)[:2000]/255.   # 维度(2000, 28, 28)-->(2000, 1, 28, 28), 矩阵每个值都/255成为(0,1)的值。  #但是验证集的x为什么要处理呢？训练集也没处理啊。
test_y = test_data.test_labels[:2000]


# 定义CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义第一个卷积及池化层
        self.conv1 = nn.Sequential(         # 每个样本输入维度 (1, 28, 28) # 1是通道数，因为是灰度图
            # Conv2d就是卷积核
            nn.Conv2d(
                in_channels=1,              # 输入通道数
                out_channels=16,            # 输出通道数（=卷积核个数）
                kernel_size=5,              # 卷积核大小
                stride=1,                   # 步长
                padding=2,                  # 填充padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)  #16是通道数，因为16个卷积核
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    # 最大池化（2x2） 每个样本的输出维度(16, 14, 14)
        )
        # 定义第二个卷积及池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # input shape (16, 14, 14) output shape (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),                # output shape (32, 7, 7) # 32是通道数
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # x.view(x.size(0), -1) 是将原来的三维矩阵拉成一个向量，因为全连接层不能连接矩阵。
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
# print(cnn)  # 打印cnn的网络结构

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # 分类用的损失函数--交叉熵损失函数


# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()


# training and testing
for epoch in range(EPOCH):          # 这个迭代是迭代整个数据集
    for step, (b_x, b_y) in enumerate(train_loader):   # 这个迭代是迭代整个batch

        output = cnn(b_x)[0]            # cnn output
        print("a"*30,output.size(),b_y.size())
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
