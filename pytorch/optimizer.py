import torch,logging
logger = logging.getLogger(__name__)


def get_optimizer(net,config):

   if 'optimizer' not in config:
      raise Exception('must include "optimizer" section in config file')

   config = config['optimizer']

   if 'name' not in config:
      raise Exception('must include "name" optimizer section in config file')

   if 'sgd' in config['name']:
      lr = config['learning_rate']
      momentum = config['momentum']
      return torch.optim.SGD(net.parameters(),lr=lr, momentum=momentum)
   elif 'adam' in config['name']:

      lr = 0.00001
      if 'lr' in config:
         lr = config['lr']

      betas = (0.9, 0.999)
      if 'betas' in config:
         betas = config['betas']

      eps = 1e-08
      if 'eps' in config:
         eps = config['eps']

      weight_decay = 0
      if 'weight_decay' in config:
         weight_decay = config['weight_decay']

      amsgrad = False
      if 'amsgrad' in config:
         amsgrad = config['amsgrad']

      return torch.optim.Adam(net.parameters(),lr,betas,eps,weight_decay,amsgrad)

   else:
      raise Exception('%s optimizer specified but not supported' % config['name'])
