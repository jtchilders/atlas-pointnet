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


def pointnet_class_loss(pred,targets,end_points,reg_weight=0.001):
   criterion = torch.nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
   classify_loss = criterion(pred, targets)
   # print('prediction = %s' % torch.nn.Softmax()(pred) )
   
   # Enforce the transformation as orthogonal matrix
   mat_loss = 0
   # if 'input_trans' in end_points:
   #    tran = end_points['input_trans']

   #    diff = torch.mean(torch.bmm(tran, tran.permute(0, 2, 1)), 0)
   #    mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1]))

   if 'feature_trans' in end_points:
      tran = end_points['feature_trans']

      diff = torch.mean(torch.bmm(tran, tran.permute(0, 2, 1)), 0)
      mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1]))

   # print('criterion = %s mat_loss = %s' % (classify_loss.item(),mat_loss.item()))
   loss = classify_loss + mat_loss * 0.001

   return loss


def multiclass_acc(pred,targets):

   # logger.info('>> pred = %s targets = %s',pred,targets)
   pred = torch.softmax(pred,dim=1)
   # logger.info('softmax = %s',pred)
   pred = pred.gt(0.4).float()
   # logger.info('gt = %s',pred)
   pred = pred.argmax(dim=1)
   # logger.info('argmax = %s',pred)

   eq = torch.eq(pred,targets)
   # logger.info('eq = %s',eq)

   return torch.sum(eq).float() / float(targets.shape[0])
