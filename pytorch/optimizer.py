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


def get_scheduler(opt,config):

   if 'optimizer' not in config:
      raise Exception('must include "loss" section in config file')

   config = config['optimizer']

   if 'lrsched' not in config:
      raise Exception('must include "lrsched" loss section in config file')

   if 'StepLR' in config['lrsched']:

      if 'lrsched_step_size' in config:
         step_size = config['lrsched_step_size']
      else:
         raise Exception('trying to use StepLR scheduler, but no step size defined in config')
      gamma = 0.1
      if 'lrsched_gamma' in config:
         gamma = config['lrsched_gamma']
      last_epoch = -1
      if 'lrsched_last_epoch' in config:
         last_epoch = config['lrsched_last_epoch']

      return torch.optim.lr_scheduler.StepLR(opt,step_size,gamma,last_epoch)
   else:
      raise Exception('%s learning rate scheduler is not recognized' % config['lrsched'])
