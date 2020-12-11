"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.1.11
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # 1~10取十个数
y = torch.linspace(10, 1, 10)       # 10~1取十个数

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # 设置大小
    shuffle=True,               # 是否打乱重排
    num_workers=2,              # subprocesses for loading data
)


def show_batch():
    for epoch in range(3):   # epoch：“代”，就是梯度下降的次数
        for step, (batch_x, batch_y) in enumerate(loader):  # step是每个mini-match的索引0 1 2 3 4
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())
            """
            Epoch:  0 | Step:  0 | batch x:  [ 5.  7. 10.  3.  4.] | batch y:  [6. 4. 1. 8. 7.]
            Epoch:  0 | Step:  1 | batch x:  [2. 1. 8. 9. 6.] | batch y:  [ 9. 10.  3.  2.  5.]
            Epoch:  1 | Step:  0 | batch x:  [ 4.  6.  7. 10.  8.] | batch y:  [7. 5. 4. 1. 3.]
            Epoch:  1 | Step:  1 | batch x:  [5. 3. 2. 1. 9.] | batch y:  [ 6.  8.  9. 10.  2.]
            Epoch:  2 | Step:  0 | batch x:  [ 4.  2.  5.  6. 10.] | batch y:  [7. 9. 6. 5. 1.]
            Epoch:  2 | Step:  1 | batch x:  [3. 9. 1. 8. 7.] | batch y:  [ 8.  2. 10.  3.  4.]
            """




if __name__ == '__main__':          #意思是当模块被直接运行时，以下代码块将被运行
    show_batch()

