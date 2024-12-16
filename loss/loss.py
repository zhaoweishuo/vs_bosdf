import torch.nn as nn
import torch
import torch.nn.functional as F
import math

"""My Loss Function"""


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.iou = IoULoss()
        self.df = DF()
        self.mse = nn.MSELoss()

    def forward(self, yhat, yhat_image, label_y, label_image):
        iou = self.iou(yhat_image, label_image)
        l1 = self.l1(yhat, label_y)
        df = self.df(yhat, label_y)
        mse = self.mse(yhat, label_y)
        dfiou = iou*(1+df.item())
        # print("l1:{:.4f} dfiou:{:.4f} df:{:.4f} mse:{:.4f} iou:{:.4f}".format(l1, dfiou, df, mse, iou))
        total = dfiou + l1

        """
        
        with open("./log/log_5000_all.txt", 'a') as file_object:
            file_object.write(
                "{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
                    l1.item(),
                    mse.item(),
                    iou.item(),
                    total.item(),
                ))
        """

        return total


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, pred, target, eps=1e-6):
        inter = torch.min(pred, target).sum()
        union = torch.sum(pred) + torch.sum(target) - inter
        iou = inter / (union + eps)
        return 1 - iou


class DF(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DF, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):

        mse = self.mse(pred, target)
        df = mse*-1
        df = df+1
        return df

