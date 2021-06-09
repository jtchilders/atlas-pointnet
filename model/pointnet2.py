import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
import logging
logger = logging.getLogger(__name__)

def get_model(config):
   net = PointNet2(config['data']['num_classes'],config['data']['num_features'])
   return net


# taken from
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch
class PointNet2(nn.Module):
   def __init__(self, num_classes, num_features):
      super(PointNet2, self).__init__()
      self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=num_features, mlp=[64, 64, 128], group_all=False)
      self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
      self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
      self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
      self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
      self.fp1 = PointNetFeaturePropagation(in_channel=128+num_features, mlp=[128, 128, 128])
      self.conv1 = nn.Conv1d(128, 128, 1)
      self.bn1 = nn.BatchNorm1d(128)
      self.drop1 = nn.Dropout(0.5)
      self.conv2 = nn.Conv1d(128, num_classes, 1)

   def forward(self, xyz):
      """
      Input:
        xyz: source points, [B, N, C], where C = 6
        cls_label: 
      Output:
        
   """
      # Set Abstraction layers
      # xyz = xyz.permute(0,2,1)

      B,N,C = xyz.shape
      
      l0_points = xyz
      l0_xyz = xyz[:,:,:3]
      
      logger.info(f'l0_xyz.shape={l0_xyz.shape} l0_points.shape={l0_points.shape}')
      l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
      logger.info(f'l1_xyz.shape={l1_xyz.shape} l1_points.shape={l1_points.shape}')
      l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
      logger.info(f'l2_xyz.shape={l2_xyz.shape} l2_points.shape={l2_points.shape}')
      l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
      logger.info(f'l3_xyz.shape={l3_xyz.shape} l3_points.shape={l3_points.shape}')
      # Feature Propagation layers
      l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
      l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
      l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],-1), l1_points)
      # FC layers
      feat =  F.relu(self.bn1(self.conv1(l0_points)))
      x = self.drop1(feat)
      x = self.conv2(x)
      x = F.log_softmax(x, dim=-1)
      # x = x.permute(0, 2, 1)
      return x, l3_points


