import torch,logging
import numpy as np
logger = logging.getLogger(__name__)


class_ids = []
def get_loss(config):
   global class_ids
   class_ids = config['data_handling']['class_nums']

   if 'loss' not in config:
      raise Exception('must include "loss" section in config file')

   loss_config = config['loss']

   if 'func' not in loss_config:
      raise Exception('must include "func" loss section in config file')

   if loss_config['func'] in globals():
      return globals()[loss_config['func']]
   elif 'CrossEntropyLoss' in loss_config['func']:
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
   
   else:
      raise Exception('%s loss function is not recognized; locals = %s' % (loss_config['func'],globals()))


def get_accuracy(config):
   if 'CrossEntropyLoss' in config['loss']['func'] or 'pointnet_class_loss' in config['loss']['func']:
      
      return multiclass_acc
   if config['loss']['acc'] in globals():
      return globals()[config['loss']['acc']]
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


def mean_class_iou(pred,targets,device='cpu'):

   nclasses = pred.shape[1]
   npoints = targets.shape[1]
   nbatch = targets.shape[0]

   targets_onehot = torch.zeros(nbatch,nclasses,npoints,device=device,requires_grad=False)
   targets_onehot = targets_onehot.scatter_(1,targets.view(nbatch,1,npoints).long(),1).float()

   pred = torch.nn.functional.softmax(pred,dim=1)

   iou = IoU_coeff(pred,targets_onehot,device=device)
   # logger.info('iou = %s',iou)

   return iou


def mean_class_iou_binary(pred,targets,device='cpu'):

   pred = pred.view(targets.shape)
   pred = torch.sigmoid(pred)

   iou = IoU_coeff_binary(pred,targets.float(),device=device)

   return iou


def IoU_coeff(pred,targets,smooth=1,device='cpu'):
   # logger.info(' pred = %s targets = %s',pred.shape,targets.shape)
   intersection = torch.abs(targets * pred).sum(dim=2)
   # logger.info(' intersection = %s ',intersection)
   union = targets.sum(dim=2) + pred.sum(dim=2) - intersection
   # logger.info(' union = %s ',union)
   iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
   # logger.info(' iou = %s ',iou)
   return iou


def IoU_coeff_binary(pred,targets,smooth=1,device='cpu'):
   # logger.info(' pred = %s targets = %s',pred.shape,targets.shape)
   intersection = torch.abs(targets * pred).sum(dim=1)
   # logger.info(' intersection = %s ',intersection)
   union = targets.sum(dim=1) + pred.sum(dim=1) - intersection
   # logger.info(' union = %s ',union)
   iou = torch.mean((intersection + smooth) / (union + smooth), dim=0)
   # logger.info(' iou = %s ',iou)
   return iou


def dice_coef(pred,targets,smooth=1,device='cpu'):
   intersection = (targets * pred).sum(dim=2)
   union = targets.sum(dim=2) + pred.sum(dim=2)
   dice = torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)
   logger.info(' dice = %s ',dice)
   return dice.mean()


def pixel_wise_cross_entry(pred,targets,endpoints,device='cpu'):

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]

   classify_loss = torch.nn.CrossEntropyLoss()(pred,targets.long())

   return classify_loss


def pixelwise_crossentropy_focalA(pred,targets,endpoints,weights,device='cpu',gamma=2.,alpha=1.):

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # weights.shape = [N_batch,N_points]

   nclasses = pred.shape[1]
   npoints = targets.shape[1]
   nbatch = targets.shape[0]

   # expand weights to [N_batch, N_class, N_points]
   weights_onehot = weights.view(nbatch,1,npoints).repeat(1,nclasses,1)
   logger.info('weights_onehot = %s',weights_onehot[0,...,0])

   targets_onehot = torch.LongTensor(nbatch,nclasses,npoints).zero_()
   targets_onehot = targets_onehot.scatter_(1,targets.view(nbatch,1,npoints).long(),1).float()
   logger.info('targets = %s',targets[0,0])
   logger.info('targets_onehot = %s',targets_onehot[0,...,0])

   model_out = -torch.nn.LogSoftmax(dim=1)(pred)  # [N_batch, N_class, N_points]
   logger.info('model_out = %s',model_out[0,...,0])
   ce = model_out * targets_onehot  # [N_batch, N_class, N_points]
   logger.info('ce = %s',ce[0,...,0])

   focal_weight = targets_onehot * torch.pow(1. - model_out,gamma)
   logger.info('focal_weight = %s',focal_weight[0,...,0])

   focal_loss = focal_weight * alpha * ce * weights_onehot
   logger.info('focal_loss = %s',focal_loss[0,...,0])

   proportional_weights = []
   for i in range(len(class_ids)):
      proportional_weights.append((targets == i).sum())
   proportional_weights = torch.Tensor(proportional_weights)
   proportional_weights = proportional_weights.sum() / proportional_weights
   proportional_weights[proportional_weights == float('Inf')] = 0

   proportional_weights = proportional_weights.view(nclasses,1).repeat(1,npoints)
   proportional_weights = proportional_weights.view(1,nclasses,npoints).repeat(nbatch,1,1)

   loss = focal_loss * proportional_weights

   loss_value = torch.mean(loss)

   return loss_value


def pixelwise_crossentropy_focal(pred,targets,endpoints,weights,device='cpu',gamma=2.,alpha=1.):
   # from https://github.com/clcarwin/focal_loss_pytorch
   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # weights.shape = [N_batch,N_points]

   nclasses = pred.shape[1]
   # npoints = targets.shape[1]
   # nbatch = targets.shape[0]

   # logger.info('targets = %s',targets[0,0])
   # logger.info('pred = %s',pred[0,...,0])
   pred = pred.transpose(1,2)  # [N_batch,N_points,N_class]
   # logger.info('pred = %s',pred[0,0,...])
   pred = pred.contiguous().view(-1,nclasses)  # [N_batch*N_points,N_class]
   # logger.info('pred = %s',pred[0])

   # targets = targets.view(-1,1).long()  # [N_batch*N_points,1]
   # logger.info('targets = %s',targets.shape)
   weights = weights.view(-1)

   logpt = torch.nn.functional.log_softmax(pred,dim=1)  # [N_batch*N_points,N_class]
   # logger.info('logpt = %s',logpt[0])
   logpt = logpt.gather(1,targets.view(-1,1).long())  # [N_batch*N_points,1]
   # logger.info('logpt = %s',logpt[0])
   logpt = logpt.view(-1)
   # logger.info('logpt = %s',logpt[0])
   pt = torch.autograd.Variable(logpt.data.exp())
   # logger.info('pt = %s',pt[0])

   loss = -1 * (1 - pt) ** gamma * logpt * weights

   return loss.mean()


def two_step_loss(pred,targets,endpoints,weights=None,device='cpu'):

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # weights.shape = [N_batch,N_points]

   # first step, calculate loss of something vs nothing

   # this is a custom loss for when using classes:
   # ["none","jet","electron","muon","tau"]

   # we will calculate the sigmoid of the "none" class and use this
   # as a something vs nothing classification

   # apply sigmoid to none class to treat as identifier of nothing
   pred_nothing = pred[:,0,:]
   targets_nothing = (targets == 0).float()
   loss_nothing = torch.nn.functional.binary_cross_entropy_with_logits(pred_nothing,targets_nothing,reduction='none') * weights
   loss_nothing = loss_nothing.sum() / weights.sum()

   # calculate the softmax for the remaining class objects, then mask based on truth
   pred_something = pred[:,1:,:]
   targets_weights = (targets > 0).float()
   targets_something = (targets - 1).float() * targets_weights
   loss_something = torch.nn.functional.cross_entropy(pred_something,targets_something.long(),reduction='none')
   loss_something = loss_something * targets_weights * weights
   loss_something = loss_something.sum() / loss_something.nonzero().shape[0]
   
   return loss_nothing + loss_something


def pixelwise_crossentropy_weighted(pred,targets,endpoints,weights=None,device='cpu',loss_offset=0.):
   # for semantic segmentation, need to compare class
   # prediction for each point AND need to weight by the
   # number of pixels for each point

   # flatten targets and predictions

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # logger.info(f'pred = {pred.shape}  targets = {targets.shape}')

   proportional_weights = []
   for i in range(len(class_ids)):
      proportional_weights.append((targets == i).sum())

   proportional_weights = torch.Tensor(proportional_weights,device=device)
   proportional_weights = proportional_weights / proportional_weights.sum()
   proportional_weights = 1 - proportional_weights
   proportional_weights[proportional_weights == float('Inf')] = 0

   loss_value = torch.nn.CrossEntropyLoss(weight=proportional_weights,reduction='none')(pred,targets.long())
   # weights zero suppresses the loss calculated from padded points which were added to make fixed length inputs
   loss_value = loss_value * weights

   return loss_value.mean() + loss_offset


def pixelwise_bce_weighted_somenone(pred,targets,endpoints,weights=None,device='cpu'):
   # for semantic segmentation, need to compare class
   # prediction for each point AND need to weight by the
   # number of pixels for each point

   # flatten targets and predictions

   # pred.shape = [N_batch, N_class, N_points]
   # targets.shape = [N_batch,N_points]
   # logger.info(f'pred = {pred.shape}  targets = {targets.shape}')
   # logger.info(f'pred = {pred}  targets = {targets}')
   
   fraction_something_classes = (weights.sum() / targets.sum()) - 1.
   pos_targets = targets.sum()
   pos_pred = (torch.sigmoid(pred) > 0.5).sum()
   logger.info('pos_targets -> %s pos_pred -> %s ',pos_targets,pos_pred)

   loss_value = torch.nn.functional.binary_cross_entropy_with_logits(pred.view(targets.shape),targets.float(),reduction='none',pos_weight=fraction_something_classes)
   # weights zero suppresses the loss calculated from padded points which were added to make fixed length inputs
   loss_value = loss_value * weights

   return loss_value.mean()
