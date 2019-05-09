import torch
import pytorch.utils as utils
import logging
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
   def __init__(self,nPoints,nCoords,nClasses):
      super(PointNet1d,self).__init__()

      self.input_trans = Transform1d(nPoints,nCoords)
      
      self.conv64A = utils.Conv1d(nCoords,64,pool=False)
      self.conv64B = utils.Conv1d(64,64,pool=False)
      
      self.feature_trans = Transform1d(nPoints,64)
      
      self.conv64C = utils.Conv1d(64,64,pool=False)
      self.conv128  = utils.Conv1d(64,128,pool=False)
      self.conv1024 = utils.Conv1d(128,1024,pool=False)
      
      self.pool = torch.nn.MaxPool1d(nPoints)
      
      self.linear512 = utils.Linear(1024,512)
      self.dropoutA = torch.nn.Dropout(0.7)
      
      self.linear256 = utils.Linear(512,256)
      self.dropoutB = torch.nn.Dropout(0.7)
   
      self.linearID = utils.Linear(256,nClasses,bn=False,activation=None)
      
   def forward(self,x):
      batch_size = x.shape[0]
      # l=10
      # print('pointnet1d input: %s' % x[0,:l,:])
      it = self.input_trans(x)
      # print('pointnet1d input_trans: %s' % it[0])
      endpoints = {'input_trans':it}
      x = torch.bmm(it,x)
      # print('pointnet1d it*x: %s' % x[0,:l,:l])
      
      x = self.conv64A(x)
      # print('pointnet1d conv64A: %s' % x[0,:l,:l])
      x = self.conv64B(x)
      # print('pointnet1d conv64B: %s' % x[0,:l,:l])
      
      ft = self.feature_trans(x)
      # print('pointnet1d feature_trans: %s' % ft[0,:10,:10])
      endpoints['feature_trans'] = ft
      x = torch.bmm(ft,x)
      # print('pointnet1d ft*x: %s' % x[0,:l,:l])
      
      x = self.conv64C(x)
      # print('pointnet1d conv64C: %s' % x[0,:l,:l])
      x = self.conv128(x)
      # print('pointnet1d conv128: %s' % x[0,:l,:l])
      x = self.conv1024(x)
      # print('pointnet1d conv1024: %s' % x[0,:l,:l])
      
      x = self.pool(x)
      # print('pointnet1d pool: %s' % x[0,:l,:])
      
      x = x.reshape([batch_size,-1])
      # print('pointnet1d reshape: %s' % x[0,:l])
      
      x = self.linear512(x)
      # print('pointnet1d linear512: %s' % x[0,:l])
      x = self.dropoutA(x)
      # print('pointnet1d dropoutA: %s' % x[0,:l])
      
      x = self.linear256(x)
      # print('pointnet1d linear256: %s' % x[0,:l])
      x = self.dropoutB(x)
      # print('pointnet1d dropoutB: %s' % x[0,:l])
      
      x = self.linearID(x)
      # print('pointnet1d linearID: %s' % x)
      
      return x,endpoints


class Transform1d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width):
      super(Transform1d,self).__init__()

      self.width = width

      self.conv64 = utils.Conv1d(width,64,pool=False)
      self.conv128 = utils.Conv1d(64,128,pool=False)
      self.conv1024 = utils.Conv1d(128,1024,pool=False)
      
      self.pool = torch.nn.MaxPool1d(height)
      
      self.linear512 = utils.Linear(1024,512)
      self.linear256 = utils.Linear(512,256)

      self.linearK = torch.nn.Linear(256,width*width)
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
