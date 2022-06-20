from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math


# add GeM layer as Global Descriptor Encoding Layer
class GeM(nn.Module):
    def __init__(self, p= 3, eps= 1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

        # self.__gem = nn.AdaptiveAvgPool2d(output_size=(self.output_dim, 1))
    
    def gem(self, x: torch.Tensor, p= 3, eps= 1e-6) -> torch.Tensor:
        # 但是不能判断是不是avg_pool2D
        # return self.__gem(x.clamp(min= eps).pow(p)).pow(1.0/p)
        return F.avg_pool2d(x.clamp(min= eps).pow(p), (1, x.size(-1))).pow(1.0/p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.transpose(1, 3).contiguous()
        # x = x.transpose(1, 3)
        
        # 变为[batch_nu, 4096, 1024]，这里是对齐 PointNet 的输出
        x = x.squeeze(3)
        
        # 变为 [batch_num, output_dim, 1]
        __res = self.gem(x, p= self.p, eps= self.eps)
        
        # 去掉最后那个 1 维
        return __res.squeeze(-1)
    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 256, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        #x = x.transpose(2,1)
        #x = torch.bmm(x, trans)
        #x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x)
            x = torch.squeeze(x)
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans)
            x = x.transpose(1, 2).contiguous()
            x = x.view(batchsize, 64, -1, 1)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.bn5(self.conv5(x))
        if not self.max_pool:
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetGeM(nn.Module):
    def __init__(self, num_points=4096, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(PointNetGeM, self).__init__()
        self.point_net = PointNetfeat(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        self.output_dim = output_dim
        # 先简单写一下
        self.pooling = GeM()

    def forward(self, x: torch.Tensor):
#         print("Before: ", x.shape)
        x = self.point_net(x)
        # x = x.squeeze(-1)
#         print("After PointNet", x.shape)
        x = self.pooling(x)
        # x = x.squeeze(-1)
        # print("After GeM:", x.shape)
        return x


if __name__ == '__main__':
    num_points = 4096
    sim_data = Variable(torch.rand(16, num_points, 3))
    sim_data = sim_data.unsqueeze(1)
    sim_data = sim_data.cuda()
    print(sim_data.shape)
    print(torch.cuda.is_available())
    png = PointNetGeM(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    png.train()
    out3 = png(sim_data)
    print('png', out3.size())
