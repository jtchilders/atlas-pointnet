import torch,logging
logger = logging.getLogger(__name__)


def get_loss(config):

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   config = config['loss']

   if 'func' not in config:
      raise Exception('must include "func" loss section in config file')

   if 'CrossEntropyLoss' in config['func']:
      weight = None
      if 'weight' in config:
         weight = config['loss_weight']
      size_average = None
      if 'size_average' in config:
         size_average = config['loss_size_average']
      ignore_index = -100
      if 'ignore_index' in config:
         ignore_index = config['loss_ignore_index']
      reduce = None
      if 'reduce' in config:
         reduce = config['loss_reduce']
      reduction = 'mean'
      if 'reduction' in config:
         reduction = config['loss_reduction']

      return torch.nn.CrossEntropyLoss(weight,size_average,ignore_index,reduce,reduction)
   if 'pointnet_class_loss' in config['func']:
      return pointnet_class_loss
   else:
      raise Exception('%s loss function is not recognized' % config['func'])


def get_accuracy(config):
   if 'CrossEntropyLoss' in config['loss']['func'] or 'pointnet_class_loss' in config['loss']['func']:
      
      return multiclass_acc
   else:
      if 'func' not in config['model']:
         raise Exception('loss function not defined in config')
      else:
         raise Exception('%s loss function is not recognized' % config['loss']['func'])


def get_scheduler(opt,config):

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   config = config['loss']

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


def pointnet_class_loss(pred,targets,end_points,reg_weight=0.001):
   criterion = torch.nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
   classify_loss = criterion(pred, targets)

   # Enforce the transformation as orthogonal matrix
   mat_loss = 0
   if 'input_trans' in end_points:
      tran = end_points['input_trans']

      diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
      mat_loss += torch.nn.MSELoss()(diff, torch.eye(3))

   if 'feature_trans' in end_points:
      tran = end_points['feature_trans']

      diff = torch.mean(torch.matmul(tran, tran.permute(0, 2, 1)), 0)
      mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1]))

   loss = classify_loss + mat_loss * 0.001

   return loss


def multiclass_acc(pred,targets):

   # logger.info('>> pred = %s targets = %s',pred,targets)
   pred = torch.softmax(pred,dim=1)
   # logger.info('softmax = %s',pred)
   pred = pred.gt(0.5).float()
   # logger.info('gt = %s',pred)
   pred = pred.argmax(dim=1)
   # logger.info('argmax = %s',pred)

   eq = torch.eq(pred,targets)
   # logger.info('eq = %s',eq)

   return torch.sum(eq).float() / float(targets.shape[0])

