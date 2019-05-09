import torch
import logging
logger = logging.getLogger(__name__)


class Conv2d(torch.nn.Module):
   def __init__(self,nIn,nOut,
                kernel_size=(3,3),stride=1,padding=0,dilation=1,groups=1,bias=True,
                bn=True,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True,
                activation='relu',
                pool=True,ptype='max',pool_size=(2,2),pool_stride=None,pool_padding=0,
                pool_dialation=1,return_indices=False,ceil_mode=False,count_include_pad=True,
                layer_number=0):
      super(Conv2d,self).__init__()

      self.conv = torch.nn.Conv2d(nIn,nOut,kernel_size,
                                  stride=stride,padding=padding,
                                  dilation=dilation,groups=groups,bias=bias)
      
      self.bn = bn
      if self.bn:
         self.bnorm = torch.nn.BatchNorm2d(nOut,eps,momentum,affine,track_running_stats)

      if activation is not None:
         self.do_activation = True
         if 'relu' in activation:
            self.activation = torch.nn.ReLU()
      else:
         self.do_activation = False

      self.do_pool = pool
      if self.do_pool:
         if 'max' in ptype:
            self.pool = torch.nn.MaxPool2d(pool_size,pool_stride,pool_padding,pool_dialation,
                                           return_indices, ceil_mode)

         elif 'avg' in ptype:
            self.pool = torch.nn.AvgPool2d(pool_size,pool_stride,pool_padding,pool_dialation,
                                           ceil_mode,count_include_pad)

   def forward(self,x):
      x = self.conv(x)
      if self.bn:
         x = self.bnorm(x)
      if self.do_activation:
         x = self.activation(x)
      if self.do_pool:
         x = self.pool(x)
      return x


class Conv1d(torch.nn.Module):
   def __init__(self,nIn,nOut,
                kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,
                bn=True,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True,
                activation='relu',
                pool=True,ptype='max',pool_size=(2,2),pool_stride=None,pool_padding=0,
                pool_dialation=1,return_indices=False,ceil_mode=False,count_include_pad=True,
                layer_number=0):
      super(Conv1d,self).__init__()

      self.conv = torch.nn.Conv1d(nIn,nOut,kernel_size,
                                  stride=stride,padding=padding,
                                  dilation=dilation,groups=groups,bias=bias)
      
      self.bn = bn
      if self.bn:
         self.bnorm = torch.nn.BatchNorm1d(nOut,eps,momentum,affine,track_running_stats)

      if activation is not None:
         self.do_activation = True
         if 'relu' in activation:
            self.activation = torch.nn.ReLU()
      else:
         self.do_activation = False

      self.do_pool = pool
      if self.do_pool:
         if 'max' in ptype:
            self.pool = torch.nn.MaxPool1d(pool_size,pool_stride,pool_padding,pool_dialation,
                                           return_indices, ceil_mode)

         elif 'avg' in ptype:
            self.pool = torch.nn.AvgPool1d(pool_size,pool_stride,pool_padding,pool_dialation,
                                           ceil_mode,count_include_pad)

   def forward(self,x):
      x = self.conv(x)
      if self.bn:
         x = self.bnorm(x)
      if self.do_activation:
         x = self.activation(x)
      if self.do_pool:
         x = self.pool(x)
      return x


class Linear(torch.nn.Module):
   def __init__(self,nIn,nOut,bias=True,
                bn=True,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True,
                activation='relu'):
      super(Linear,self).__init__()

      self.linear = torch.nn.Linear(nIn,nOut,bias)

      self.bn = bn
      if self.bn:
         self.bnorm = torch.nn.BatchNorm1d(nOut,eps,momentum,affine,track_running_stats)

      if activation is not None:
         self.do_activation = True
         if 'relu' in activation:
            self.activation = torch.nn.ReLU()
      else:
         self.do_activation = False

   def forward(self,x):
      x = self.linear(x)
      if self.bn:
         x = self.bnorm(x)
      if self.do_activation:
         x = self.activation(x)
      return x
