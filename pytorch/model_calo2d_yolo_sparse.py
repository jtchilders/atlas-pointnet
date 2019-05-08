import torch.nn as nn
import torch
import logging
import sparseconvnet as scn
logger = logging.getLogger(__name__)


class FullBlock2D(nn.Module):
   def __init__(self,nIn,nOut,filter_size=(3,3),pool_size=(2,2),layer_number=0):
      super(FullBlock2D,self).__init__()
      self.conv = Conv2D(nIn,nOut,filter_size)
      self.bnorm_lrelu = BatchNormLeakyReLU(nOut)
      self.pool = MaxPool2D(pool_size)

      self.add_module('conv2d_%2d' % layer_number,self.conv)
      self.add_module('bnorm_lrelu_%2d' % layer_number,self.bnorm_lrelu)
      self.add_module('max_pool_%2d' % layer_number,self.pool)

   def forward(self,x):
      x = self.conv(x)
      x = self.bnorm_lrelu(x)
      return self.pool(x)


class PartBlock2D(nn.Module):
   def __init__(self,nIn,nOut,filter_size=(3,3),layer_number=0):
      super(PartBlock2D,self).__init__()
      self.conv = Conv2D(nIn,nOut,filter_size)
      self.bnorm_lrelu = BatchNormLeakyReLU(nOut)

      self.add_module('conv2d_%2d' % layer_number,self.conv)
      self.add_module('bnorm_lrelu_%2d' % layer_number,self.bnorm_lrelu)

   def forward(self,x):
      x = self.conv(x)
      return self.bnorm_lrelu(x)


class DensePartBlock2D(nn.Module):
   def __init__(self,nIn,nOut,filter_size=(3,3),layer_number=0):
      super(DensePartBlock2D,self).__init__()
      self.conv = nn.Conv2d(nIn,nOut,filter_size,stride=1, padding=0, dilation=1, groups=1, bias=False)
      self.bnorm = nn.BatchNorm2d(nOut)
      self.lrelu = nn.LeakyReLU()

      self.add_module('conv2d_%2d' % layer_number,self.conv)
      self.add_module('bnorm_%2d' % layer_number,self.bnorm)
      self.add_module('lrelu_%2d' % layer_number,self.lrelu)

   def forward(self,x):
      x = self.conv(x)
      x = self.bnorm(x)
      return self.lrelu(x)


class BlockSeries2D(nn.Module):
   def __init__(self,block_pars,layer_number=0):
      super(BlockSeries2D,self).__init__()

      self.blocks = []

      # block_pars is a list of dictionaries
      for block_par in block_pars:
         block = None
         if block_par['type'] == 'full':
            block = FullBlock2D
         else:
            block = PartBlock2D
         block_par['opts']['layer_number'] = layer_number
         self.blocks.append(block(**block_par['opts']))
         self.add_module('%sblock_%03d' % (block_par['type'],layer_number),self.blocks[-1])
         layer_number += 1

   def forward(self,x):

      for block in self.blocks:
         # logger.info('output from block: %s',block)
         x = block(x)

      return x


class Net2D(nn.Module):
   def __init__(self, input_spatial_shape=[256, 5760], input_channels=2):
      super(Net2D,self).__init__()

      layer_number = 0
      self.input_layer = scn.InputLayer(input_channels,input_spatial_shape)

      self.pooling_factor = [1,1]

      self.block_pars_preconnect = [
         {'type':'full','opts':{'nIn':   2,'nOut':  32,'filter_size':(3,3),'pool_size':(1,2)}},  # 1
         {'type':'full','opts':{'nIn':  32,'nOut':  64,'filter_size':(3,3),'pool_size':(1,2)}},  # 2
         # {'type':'part','opts':{'nIn':  64,'nOut': 128,'filter_size':(3,3)}},  # 3
         # {'type':'part','opts':{'nIn': 128,'nOut':  64,'filter_size':(1,1)}},  # 4
         {'type':'full','opts':{'nIn':  64,'nOut': 128,'filter_size':(3,3),'pool_size':(1,2)}},  # 5
         # {'type':'part','opts':{'nIn': 128,'nOut': 256,'filter_size':(3,3)}},  # 6
         # {'type':'part','opts':{'nIn': 256,'nOut': 128,'filter_size':(1,1)}},  # 7
         {'type':'full','opts':{'nIn': 128,'nOut': 256,'filter_size':(3,3),'pool_size':(2,2)}},  # 8
         {'type':'full','opts':{'nIn': 256,'nOut': 512,'filter_size':(3,3),'pool_size':(2,2)}},  # 9
         {'type':'part','opts':{'nIn': 512,'nOut': 256,'filter_size':(1,1)}},  # 10
         {'type':'full','opts':{'nIn': 256,'nOut': 512,'filter_size':(3,3),'pool_size':(2,2)}},  # 11
         # {'type':'part','opts':{'nIn': 512,'nOut': 256,'filter_size':(1,1)}},  # 12
         # {'type':'full','opts':{'nIn': 256,'nOut': 512,'filter_size':(3,3),'pool_size':(2,2)}},  # 13
         {'type':'full','opts':{'nIn': 512,'nOut': 1024,'filter_size':(3,3),'pool_size':(2,2)}},  # X
         ]

      self.block_pars_postconnect = [
         # {'type':'part','opts':{'nIn': 512,'nOut':1024,'filter_size':(3,3)}},  # 14
         # {'type':'part','opts':{'nIn':1024,'nOut': 512,'filter_size':(1,1)}},  # 15
         # {'type':'part','opts':{'nIn': 512,'nOut':1024,'filter_size':(3,3)}},  # 16
         # {'type':'part','opts':{'nIn':1024,'nOut': 512,'filter_size':(1,1)}},  # 17
         # {'type':'part','opts':{'nIn': 512,'nOut':1024,'filter_size':(3,3)}},  # 18
         {'type':'part','opts':{'nIn':1024,'nOut':1024,'filter_size':(3,3)}},  # 19
         {'type':'part','opts':{'nIn':1024,'nOut':1024,'filter_size':(3,3)}},  # 20
         ]

      for layer in (self.block_pars_preconnect + self.block_pars_postconnect):
         if 'full' in layer['type']:
            pool = layer['opts']['pool_size']
            self.pooling_factor = [self.pooling_factor[i] / pool[i] for i in range(len(self.pooling_factor))]

      self.grid = [int(input_spatial_shape[i] * self.pooling_factor[i]) for i in range(len(self.pooling_factor))]
      logger.info('grid = %s',self.grid)

      layer_number = 1
      self.preconnection = BlockSeries2D(self.block_pars_preconnect, layer_number)
      layer_number += len(self.block_pars_preconnect)
      self.postconnection = BlockSeries2D(self.block_pars_postconnect,layer_number)
      layer_number += len(self.block_pars_postconnect)

      self.preconnect_sp2dn = scn.SparseToDense(2,self.block_pars_preconnect[-1]['opts']['nOut'])
      self.postconnect_sp2dn = scn.SparseToDense(2,self.block_pars_postconnect[-1]['opts']['nOut'])

      self.CombinedLayer = DensePartBlock2D(self.block_pars_postconnect[-1]['opts']['nOut'] + self.block_pars_preconnect[-1]['opts']['nOut'],1024,(1,1),layer_number)

      self.LastConv = nn.Conv2d(1024,4,(1,1))

   def forward(self,x):

      logger.debug('input_layer')
      x = self.input_layer(x)

      logger.debug('preconnection')
      connection = self.preconnection(x)

      logger.debug('postconnection')
      x = self.postconnection(connection)

      logger.debug('preconnect_sp2dn')
      connection = self.preconnect_sp2dn(connection)
      logger.debug('preconnect_sp2dn')
      x = self.postconnect_sp2dn(x)

      logger.debug('cat')
      # logger.info('shapes: connection: %s, x: %s',connection.shape,x.shape)
      x = torch.cat([x,connection],1)

      logger.debug('CombinedLayer')
      # logger.info('connected: %s',x.shape)
      x = self.CombinedLayer(x)
      # logger.info('combined: %s',x.shape)

      logger.debug('LastConv')
      x = self.LastConv(x)
      # logger.info('last: %s',x.shape)

      return x


def Conv2D(nIn, nOut,
           filter_size=(3,3), dimension=2, bias=False):
   return scn.SubmanifoldConvolution(dimension, nIn, nOut, filter_size, bias)


def BatchNormLeakyReLU(nPlanes, eps=1e-4, momentum=0.9, leakiness=0.333):
   return scn.BatchNormLeakyReLU(nPlanes,eps,momentum,leakiness)


def MaxPool2D(pool_size, pool_stride=None,dimension=2,nFeaturesToDrop=0):
   if pool_stride is None: pool_stride = pool_size
   return scn.MaxPooling(dimension, pool_size, pool_stride, nFeaturesToDrop)
