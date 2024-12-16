import torch
import loader
from torch.utils import data
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import model
import time
import loss
from torchsummary import summary

def link_sim():
    client = RemoteAPIClient()
    sim = client.require('sim')
    return sim

def get_image(sim, points):
    target_handle = sim.getObject('/AUBO_i5_Base/target')
    sensor_handle = sim.getObject('/AUBO_i5_Base/Vision_sensor')
    points = points.cpu()
    points = points.detach().numpy()
    pred_img = []

    for i in points:
        buildmatrix = sim.buildMatrix(i[0:3], i[3:6])
        sim.setObjectMatrix(target_handle, buildmatrix)
        time.sleep(0.1)
        image, resolution = sim.getVisionSensorImg(sensor_handle)
        image = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        image = np.flipud(image)
        image = np.ascontiguousarray(image)
        transform = transforms.ToTensor()
        data = transform(image)
        pred_img.append(data)
    pred_tensor = torch.stack([matrix for matrix in pred_img])
    return pred_tensor



def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()
EPOCHS = 60
BATCH_SIZE = 2
train_dataset = loader.LoadTrainDataset(data_root="./dataset/5000/train/")
train_data_loader = data.DataLoader(
                dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )
net = model.BODFNet(pretrained=True).to(DEVICE)
# loss = loss.MyLoss() # Use IoU
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS+3, eta_min=1e-7)
# sim = link_sim()  # Use IoU

"""Print model's information"""
summary(net, input_size=(3, 224, 224))


def train(model, device, train_loader, loss, optimizer, scheduler, epoch, sim):
    model.train()
    train_loss = []

    for batch_idx, data in enumerate(train_loader):

        x, label, label_image = data['image'].to(device), data['label'].to(device), data['label_image'].to(device)

        optimizer.zero_grad()
        _, y_hat = model(x)

        """
        # Use IoU
        yhat_image = get_image(sim, y_hat)
        yhat_image = yhat_image.to(device)
        l = loss(y_hat.float(), yhat_image, label.float(), label_image)
        """

        l = loss(y_hat.float(),label.float())

        l.backward()
        optimizer.step()
        train_loss.append(l.item())

        print('Train Epoch: {} [{}/{} ({:.0f}%)] batch_loss:{:.4f}'.format(
                epoch, (batch_idx+1) * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), l.item()))

    train_loss = sum(train_loss) / len(train_loss)
    print("Epoch: {} train loss:{:.4f}".format(epoch, train_loss))
    print("epoch %d lr：%f" % (epoch, optimizer.param_groups[0]['lr']))
    torch.save(net.state_dict(), "checkpoints/5000_mse.pth")
    scheduler.step()

    #  Save the log
    with open("./log/log_5000_mse.txt", 'a') as file_object:
        file_object.write(
            "{},{:.6f},{:.8f}\n".format(
                epoch,
                train_loss,
                optimizer.param_groups[0]['lr'],
            ))


if __name__ == '__main__':
    for epoch_num in range(1, EPOCHS + 1):
        train(net, DEVICE, train_data_loader, loss, optimizer, scheduler, epoch_num, None)
