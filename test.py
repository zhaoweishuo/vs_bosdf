"""
测试模型
"""
import torch
import loader
from torch.utils import data
import model
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas
import cv2
import os
import datetime

print(torch.__version__)
exit()
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


DEVICE = try_gpu()
BATCH_SIZE = 1
data_root = "./dataset/5000/train/"
test_dataset = loader.LoadTestDataset(data_root=data_root)
test_data_loader = data.DataLoader(
                dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
L1loss = torch.nn.L1Loss()


client = RemoteAPIClient()
sim = client.require('sim')
target_handle = sim.getObject('/AUBO_i5_Base/target')
sensor_handle = sim.getObject('/AUBO_i5_Base/Vision_sensor')


def fast_move_and_show_image(name, pose):
    buildmatrix = sim.buildMatrix(pose[0:3], pose[3:6])  # 移动
    sim.setObjectMatrix(target_handle, buildmatrix)
    time.sleep(1)
    image, resolution = sim.getVisionSensorImg(sensor_handle)
    image = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.flip(image, 0)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./operationInfo/"+name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    time.sleep(1)
    print("./operationInfo/"+name+" has been saved")


def fast_move_and_get_image(pose):
    buildmatrix = sim.buildMatrix(pose[0:3], pose[3:6])  # 移动
    sim.setObjectMatrix(target_handle, buildmatrix)
    time.sleep(0.5)
    image, resolution = sim.getVisionSensorImg(sensor_handle)
    image = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    image = Image.fromarray(image)
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # 不知道为什么和仿真中显示的传感器图像上下颠倒，旋转回来
    return image


def smooth_move(sim_handle, tar_handle, prediction):
    current_pose = sim_handle.getObjectPose(tar_handle)
    target_pose = sim_handle.buildPose(prediction[0:3], prediction[3:6])
    # print(currentPose)
    # print(targetPose)
    max_vel = [0.02,0.02,0.02,0.2]
    max_accel = [0.002,0.002,0.002,0.02]
    max_jerk = [0.001,0.001,0.001,0.01]
    sim_handle.moveToPose(-1, current_pose, max_vel, max_accel, max_jerk, target_pose, move_to_pose_callback, tar_handle)
    print("smooth move done")


def move_to_pose_callback(tr, vel, accel, handle):
    sim.setObjectPose(handle, tr)

    matrix = sim.poseToMatrix(tr)
    euler_angles = sim.getEulerAnglesFromMatrix(matrix)
    pose = [tr[0], tr[1], tr[2], euler_angles[0], euler_angles[1], euler_angles[2]]
    # print('pose:', pose)
    move = {
        "x": pose[0],
        "y": pose[1],
        "z": pose[2],
        "a": pose[3],
        "b": pose[4],
        "g": pose[5],
    }
    move_data = pandas.DataFrame(data=move, index=[0])  #
    move_data.to_csv("./operationInfo/pose.csv", mode='a', index=False, header=False)  # 保存数据

    linear_velocity, angular_velocity = sim.getObjectVelocity(handle)
    # print('velocity:', linear_velocity, angular_velocity)
    speed = {
        "x": linear_velocity[0],
        "y": linear_velocity[1],
        "z": linear_velocity[2],
        "a": angular_velocity[0],
        "b": angular_velocity[1],
        "g": angular_velocity[2]
    }
    speed_data = pandas.DataFrame(data=speed, index=[0])  #
    speed_data.to_csv("./operationInfo/velocity.csv", mode='a', index=False, header=False)

    pass


def save_nodes(nodes, path):
    nodes = np.squeeze(nodes)
    header = list(range(0, 1000, 1))
    nodes_data = pandas.DataFrame(columns=header)
    nodes_data.loc[0] = nodes
    nodes_data.to_csv(path, mode='a', index=False, header=False)
    print("save nodes done")


def test(model, bls, device, test_loader):
    model.eval()
    net_loss = []
    bls_loss = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            x, y = data['image'].to(device), data['label'].to(device)
            nodes, y_hat = model(x)
            ly = L1loss(y_hat, y)

            name = data['name'][0]
            prediction = y_hat[0].cpu().numpy()
            label = data['label'][0].numpy()


            # save nodes for bls
            # nodes = nodes.cpu().numpy()
            # save_nodes(nodes, "./dataset/5000/nodes.csv")

            bls_y = bls(nodes)
            lbls = L1loss(bls_y, y)
            prediction2 = bls_y[0].cpu().numpy()

            print("image name:", name)
            print("pred:", prediction)
            print("pred2:", prediction2)
            print("gt  :", label)
            print("L1y loss:", ly.item())
            print("L1bls loss:", lbls.item())
            net_loss.append(ly.item())
            bls_loss.append(lbls.item())

            image = fast_move_and_get_image(prediction)
            image2 = Image.open(data_root + 'image/' + data['path'][0])
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.gca().set_title("pred")
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.gca().set_title("gt")
            plt.imshow(image2)
            plt.show()

    net_loss = sum(net_loss) / len(net_loss)
    bls_loss = sum(bls_loss) / len(bls_loss)
    print("net_loss:{:.4f} bls_loss:{:.4f}".format(net_loss, bls_loss))


if __name__ == '__main__':
    net = model.BODFNet(pretrained=False).to(DEVICE)
    net.load_state_dict(torch.load("./checkpoints/5000_my.pth"))
    bls = model.Bls(6, 8000, 6,1000).to(DEVICE)
    # bls.load_state_dict(torch.load("./checkpoints/bls_my.pth"))
    test(net, bls, DEVICE, test_data_loader)

