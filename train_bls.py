import torch
import loader
from torch.utils import data
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import model
from torchsummary import summary


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()
EPOCHS = 256
BATCH_SIZE = 30
bls_dataset = loader.LoadBlsDataset(data_root="./dataset/5000/")
bls_data_loader = data.DataLoader(
                dataset=bls_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
net = model.Bls(6, 8000, 6,1000).to(DEVICE)
loss = torch.nn.L1Loss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS+3, eta_min=1e-7)
"""Print model's information"""
summary(net, input_size=(1000, ))
exit()

def train(model, device, train_loader, loss, optimizer, scheduler, epoch):
    model.train()  # 训练时调用，启用batch normalization和dropout
    train_loss = []

    for batch_idx, data in enumerate(train_loader):
        # 这里x为输入对应图片，y为图片对应的label
        x, label = data['node'].to(device), data['label'].to(device)  # 把x，y转移至指定设备
        optimizer.zero_grad()  # 梯度归零
        y_hat = model(x)  # 模型输出结果y_hat
        l = loss(y_hat.float(), label.float())  # 计算损失  使用MSE需要.float()
        l.backward()  # 反向传播计算梯度
        optimizer.step()  # 执行一次优化步骤
        train_loss.append(l.item())
        """
            (batch_idx+1) * len(x) 当前批次*批次长度=当前完成了多少个数据
            len(train_loader.dataset) 总数据量
            100. * batch_idx / len(train_loader) (当前批次索引*100/总批次量)% = 当前周期完成百分比
            loss.item() 损失函数值
        """
        print('Train Epoch: {} [{}/{} ({:.0f}%)] batch_loss:{:.4f}'.format(
                epoch, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))

    train_loss = sum(train_loss) / len(train_loss)
    print("Epoch: {} train loss:{:.4f}".format(epoch, train_loss))
    print("epoch %d lr：%f" % (epoch, optimizer.param_groups[0]['lr']))
    torch.save(net.state_dict(), "checkpoints/bls_my.pth")  # 由于训练时间过长 每一轮存一次
    scheduler.step()  # 要放在epoch循环 不要放在batch循环

    #  Save the log
    with open("./log/log_bls_my.txt", 'a') as file_object:
        file_object.write(
            "Epoch: {} loss:{:.6f} lr:{:.8f}\n".format(
                epoch,
                train_loss,
                optimizer.param_groups[0]['lr'],
            ))


if __name__ == '__main__':
    for epoch_num in range(1, EPOCHS + 1):
        train(net, DEVICE, bls_data_loader, loss, optimizer, scheduler, epoch_num)
