import torch
import pytorch.utils as utils
import logging,time
from sklearn.metrics import confusion_matrix
import CalcMean
import numpy as np
from pytorch.model import device
logger = logging.getLogger(__name__)


class PointNet2d(torch.nn.Module):
   def __init__(self,nChannels,nPoints,nCoords,nClasses):
      super(PointNet2d,self).__init__()

      self.input_trans = Transform2d(nPoints,nCoords,nChannels,3,initial_reshape=True)
      
      self.conv64A = utils.Conv2d(nChannels,64,(1,nCoords),pool=False)
      self.conv64B = utils.Conv2d(64,64,(1,1),pool=False)
      
      self.feature_trans = Transform2d(nPoints,1,64,64)
      
      self.conv64C = utils.Conv2d(64,64,(1,1),pool=False)
      self.conv128  = utils.Conv2d(64,128,(1,1),pool=False)
      self.conv1024 = utils.Conv2d(128,1024,(1,1),pool=False)
      
      self.pool = torch.nn.MaxPool2d((nPoints,1))
      
      self.linear512 = utils.Linear(1024,512)
      self.dropoutA = torch.nn.Dropout(0.7)
      
      self.linear256 = utils.Linear(512,256)
      self.dropoutB = torch.nn.Dropout(0.7)
   
      self.linearID = utils.Linear(256,nClasses,bn=False,activation=None)
      
   def forward(self,x):
      batch_size = x.shape[0]
      
      it = self.input_trans(x)
      endpoints = {'input_trans':it}
      x = torch.matmul(x,it)
      
      x = x.reshape((batch_size,1) + x.shape[1:])
      
      x = self.conv64A(x)
      x = self.conv64B(x)
      
      ft = self.feature_trans(x)
      endpoints['feature_trans'] = ft
      x = torch.matmul(ft,x.squeeze(dim=-1))
      
      x = x.reshape((batch_size,) + x.shape[1:] + (1,))
      
      x = self.conv64C(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      endpoints['global_features'] = x
      
      x = self.linear512(x)
      x = self.dropoutA(x)
      
      x = self.linear256(x)
      x = self.dropoutB(x)
      
      x = self.linearID(x)
      
      return x,endpoints


class Transform2d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width,channels,K,initial_reshape=False):
      super(Transform2d,self).__init__()

      self.K = K
      self.initial_reshape = initial_reshape

      self.conv64 = utils.Conv2d(channels,64,(1,width),pool=False)
      self.conv128 = utils.Conv2d(64,128,(1,1),pool=False)
      self.conv1024 = utils.Conv2d(128,1024,(1,1),pool=False)
      
      self.pool = torch.nn.MaxPool2d((height,1))
      
      self.linear512 = utils.Linear(1024,512)
      self.linear256 = utils.Linear(512,256)

      self.weights = torch.zeros(256,K * K,requires_grad=True)
      self.biases  = torch.eye(K,requires_grad=True).flatten()

   def forward(self,x):
      batch_size = x.shape[0]
      
      if self.initial_reshape:
         x = x.reshape((batch_size,1) + x.shape[1:])
      
      x = self.conv64(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = torch.reshape(x,[batch_size,-1])
      x = self.linear512(x)
      x = self.linear256(x)
      
      x = torch.matmul(x,self.weights) + self.biases
      x = torch.reshape(x,[batch_size,self.K,self.K])
      
      return x


class PointNet1d(torch.nn.Module):
   def __init__(self,config,bn=True):
      super(PointNet1d,self).__init__()

      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 2)

      nPoints = input_shape[0]
      nCoords = input_shape[1]
      nClasses = len(config['data_handling']['classes'])
      
      logger.debug('nPoints = %s, nCoords = %s, nClasses = %s',nPoints,nCoords,nClasses)

      self.input_trans = Transform1d(nPoints,nCoords,bn=bn)
      
      self.input_to_feature = torch.nn.Sequential()
      for x in config['model']['input_to_feature']:
         N_in,N_out,pool = x
         self.input_to_feature.add_module('conv_%d_to_%d' % (N_in,N_out),utils.Conv1d(N_in,N_out,bn=bn,pool=pool))
         #utils.Conv1d(nCoords,64,pool=False)
         #utils.Conv1d(64,64,pool=False))
      
      self.feature_trans = Transform1d(nPoints,config['model']['input_to_feature'][-1][1],bn=bn)
      
      self.feature_to_pool = torch.nn.Sequential()
      for x in config['model']['feature_to_pool']:
         N_in,N_out,pool = x
         self.feature_to_pool.add_module('conv_%d_to_%d' % (N_in,N_out),utils.Conv1d(N_in,N_out,bn=bn,pool=pool))
         
      self.pool = torch.nn.MaxPool1d(nPoints)

      self.dense_layers = torch.nn.Sequential()
      for x in config['model']['dense_layers']:
         N_in,N_out,dropout,bn,act = x
         dr = int(dropout * 3)
         if N_out is None:
            N_out = nClasses
         self.dense_layers.add_module('dense_%d_to_%d' % (N_in,N_out),utils.Linear(N_in,N_out,bn=bn,activation=act))
         if dropout > 0:
            self.dense_layers.add_module('dropout_%03d' % dr,torch.nn.Dropout(dropout))
      
   def forward(self,x):
      batch_size = x.shape[0]
      
      # logger.info(f'input = {x.shape}')
      it = self.input_trans(x)
      endpoints = {'input_trans':x}

      x = torch.bmm(it,x)
      # logger.info(f'input_trans = {x.shape}')
      
      x = self.input_to_feature(x)
      # logger.info(f'input_to_feature = {x.shape}')
      
      ft = self.feature_trans(x)
      endpoints['feature_trans'] = x

      x = torch.bmm(ft,x)
      # logger.info(f'feature_trans = {x.shape}')
      
      x = self.feature_to_pool(x)
      # logger.info(f'feature_to_pool = {x.shape}')
      
      x = self.pool(x)
      # logger.info(f'pool = {x.shape}')
      
      x = x.reshape([batch_size,-1])
      endpoints['global_features'] = x
      # logger.info(f'global_features = {x.shape}')
      
      x = self.dense_layers(x)
      # logger.info(f'dense_layers = {x.shape}')
      
      return x,endpoints


class Transform1d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width,bn=False):
      super(Transform1d,self).__init__()

      self.width = width

      self.conv64 = utils.Conv1d(width,64,bn=bn,pool=False)
      self.conv128 = utils.Conv1d(64,128,bn=bn,pool=False)
      self.conv1024 = utils.Conv1d(128,1024,bn=bn,pool=False)
      
      self.pool = torch.nn.MaxPool1d(height)
      
      self.linear512 = utils.Linear(1024,512,bn=bn)
      self.linear256 = utils.Linear(512,256,bn=bn)

      self.linearK = torch.nn.Linear(256,width * width)
      self.linearK.bias = torch.nn.Parameter(torch.eye(width).view(width * width))

      #self.weights = torch.zeros(256,width * width,requires_grad=True)
      #self.biases  = torch.eye(width,requires_grad=True).flatten()

   def forward(self,x):
      batch_size = x.shape[0]
      
      x = self.conv64(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      x = self.linear512(x)
      x = self.linear256(x)
      
      #x = torch.matmul(x,self.weights) + self.biases
      x = self.linearK(x)
      x = x.reshape([batch_size,self.width,self.width])
      
      return x


class PointNet1d_SemSeg(torch.nn.Module):

   def __init__(self,config,bn=True):
      super(PointNet1d_SemSeg,self).__init__()

      self.pointnet1d = PointNet1d(config)

      nClasses = len(config['data_handling']['classes'])
      width = 64 + 1024

      self.conv512 = utils.Conv1d(width,512,bn=bn,pool=False)
      self.conv256 = utils.Conv1d(512,256,bn=bn,pool=False)
      self.conv128 = utils.Conv1d(256,128,bn=bn,pool=False)
      self.convclass = utils.Conv1d(128,nClasses,bn=False,pool=False)

   def forward(self,x):

      image_classes,endpoints = self.pointnet1d(x)

      pointwise_features = endpoints['feature_trans']
      global_features = endpoints['global_features']
      global_features = global_features.view(global_features.shape[0],global_features.shape[1],1)
      global_features = global_features.repeat(1,1,pointwise_features.shape[-1])

      # logger.info(f'{pointwise_features.shape}')
      # logger.info(f'{global_features.shape}')

      combined = torch.cat((pointwise_features,global_features),1)

      # logger.info(f'{combined.shape}')


      x = self.conv512(combined)
      x = self.conv256(x)
      x = self.conv128(x)
      x = self.convclass(x)

      return x,endpoints
