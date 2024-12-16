import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import torch


class LoadTrainDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        csv = pd.read_csv(self.root+"/label.csv")
        self.path_list = csv.iloc[:, 0].to_numpy()  # image column
        self.label_list = csv.iloc[:, 1:7].to_numpy()  # label column
        self.resize = 224
        self.image_num = len(self.path_list)

    def __getitem__(self, item):

        img = Image.open(self.root+'image/'+self.path_list[item])
        label = self.label_list[item].astype(np.float64)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1, scale=(0.02, 0.03), value=1),

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = image_transform(img)

        label_image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
        ])
        label_image = label_image_transform(img)


        return {'image': image, 'label': label, 'label_image': label_image, 'name': self.path_list[item].split('.')[0],}

    def __len__(self):

        return self.image_num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        csv = pd.read_csv(self.root+"label.csv")
        self.path_list = csv.iloc[:, 0].to_numpy()  # image column
        self.label_list = csv.iloc[:, 1:7].to_numpy()  # label column
        self.resize = 224
        self.image_num = len(self.path_list)

    def __getitem__(self, item):

        img = Image.open(self.root+'image/'+self.path_list[item])
        label = self.label_list[item].astype(np.float64)
        size = img.size
        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = image_transform(img)

        return {
                'image': img,
                'name': self.path_list[item].split('.')[0],
                'label': label,
                'path': self.path_list[item],
                'w': str(size[0]),
                'h': str(size[1]),
                }

    def __len__(self):
        return self.image_num


class LoadBlsDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        node_csv = pd.read_csv(self.root+"nodes.csv", header=None)
        label_csv = pd.read_csv(self.root + "label.csv")
        self.node_list = node_csv.iloc[:].to_numpy()  # get column
        self.label = label_csv.iloc[:, 1:7].to_numpy()  # get column
        self.image_num = len(self.label)



    def __getitem__(self, item):

        node = self.node_list[item]
        label = self.label[item]

        node = torch.tensor(node, dtype=torch.float)

        return {
                'node': node,
                'label': label,
                }

    def __len__(self):
        return self.image_num
