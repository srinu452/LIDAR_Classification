
# =====================================
# 5) models/pointnet.py
# =====================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):  # x: [B, k, N]
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        init = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(B, 1)
        x = self.fc3(x) + init
        x = x.view(-1, self.k, self.k)
        return x

class PointNetClassifier(nn.Module):
    def __init__(self, in_dim=10, num_classes=3):
        super().__init__()
        # We apply a T-Net only on XYZ (first 3 dims), then concat with the rest features
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(in_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dp1 = nn.Dropout(0.3)
        self.dp2 = nn.Dropout(0.3)

    def forward(self, x):  # x: [B, N, F]
        B, N, Fd = x.shape
        xyz = x[:, :, :3]
        feat_rest = x[:, :, 3:]

        # Transform XYZ
        trans = self.input_transform(xyz.transpose(1, 2))  # [B,3,3]
        xyz_t = torch.bmm(xyz, trans)  # [B,N,3]
        x_all = torch.cat([xyz_t, feat_rest], dim=2)  # [B,N,F]

        x_all = x_all.transpose(1, 2)  # [B,F,N]
        x = F.relu(self.bn1(self.conv1(x_all)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=False)[0]  # [B,1024]
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.dp2(x)
        logits = self.fc3(x)  # [B,C]
        # Per-block label distribution; for point-wise, we tile logits
        return logits
