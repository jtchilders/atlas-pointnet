import torch,logging
import numpy as np
logger = logging.getLogger(__name__)


class_ids = []
def get_loss(config):

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   loss_config = config['loss']

   if 'func' not in loss_config:
      raise Exception('must include "func" loss section in config file')

   if 'CrossEntropyLoss' in loss_config['func']:
      weight = None
      if 'weight' in loss_config:
         weight = loss_config['loss_weight']
      size_average = None
      if 'size_average' in loss_config:
         size_average = loss_config['loss_size_average']
      ignore_index = -100
      if 'ignore_index' in loss_config:
         ignore_index = loss_config['loss_ignore_index']
      reduce = None
      if 'reduce' in loss_config:
         reduce = loss_config['loss_reduce']
      reduction = 'mean'
      if 'reduction' in loss_config:
         reduction = loss_config['loss_reduction']

      return torch.nn.CrossEntropyLoss(weight,size_average,ignore_index,reduce,reduction)
   if 'pointnet_class_loss' in loss_config['func']:
      return pointnet_class_loss
   elif 'pixel_wise_cross_entry' in loss_config['func']:
      global class_ids
      class_ids = config['data_handling']['class_nums']
      return pixel_wise_cross_entry
   else:
      raise Exception('%s loss function is not recognized' % loss_config['func'])


def get_accuracy(config):
   if 'CrossEntropyLoss' in config['loss']['func'] or 'pointnet_class_loss' in config['loss']['func']:
      
      return multiclass_acc
   if 'pixel_wise_cross_entry' in config['loss']['func']:
      return pixel_wise_accuracy
   else:
      if 'func' not in config['model']:
         raise Exception('loss function not defined in config')
      else:
         raise Exception('%s loss function is not recognized' % config['loss']['func'])


def pointnet_class_loss(pred,targets,end_points,reg_weight=0.001,device='cpu'):
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
      mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1],device=device))

   # print('criterion = %s mat_loss = %s' % (classify_loss.item(),mat_loss.item()))
   loss = classify_loss + mat_loss * reg_weight

   return loss


def multiclass_acc(pred,targets):

   # logger.info('>> pred = %s targets = %s',pred,targets)
   pred = torch.softmax(pred,dim=1)
   # logger.info('gt = %s',pred)
   pred = pred.argmax(dim=1).float()
   # logger.info('argmax = %s',pred)

   eq = torch.eq(pred,targets.float())
   # logger.info('eq = %s',eq)

   return torch.sum(eq).float() / float(targets.shape[0])


def pixel_wise_accuracy(pred,targets,device='cpu'):
   # need to calculate the accuracy over all points

   pred_stat = torch.nn.Softmax(dim=1)(pred)
   _,pred_value = pred_stat.max(dim=1)

   correct = (targets.long() == pred_value).sum()
   total = float(pred_value.numel())

   acc = correct.float() / total

   return acc


def pixel_wise_cross_entry(pred,targets,endpoints,device='cpu'):
   # for semantic segmentation, need to compare class
   # prediction for each point AND need to weight by the
   # number of pixels for each point

   # flatten targets and predictions

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # logger.info(f'pred = {pred.shape}  targets = {targets.shape}')

   weights = []
   for i in range(len(class_ids)):
      weights.append((targets == i).sum())
   weights = torch.Tensor(weights)
   weights = weights.sum() / weights
   weights[weights == float('Inf')] = 0

   logger.info('weights = %s',weights)

   loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(weights))

   loss_value = loss(pred,targets.long())
   logger.info(' pred[0,...,0] = %s targets[0,0] = %s loss = %s',pred[0,...,0],targets[0,0],loss_value)

   return loss_value

