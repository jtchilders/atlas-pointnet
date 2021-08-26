import torch
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)

def softmax_accuracy(y_pred,labels,weights):
   logger.debug('y_pred.shape=%s labels.shape=%s weights.shape=%s',y_pred.shape,labels.shape,weights.shape)
   # y_pred = y_pred.permute(0,2,1)
   # number of non-zero points in this batch
   nonzero = torch.sum(weights)
   logger.debug('nonzero=%s',nonzero)
   # convert logits to predictions
   y_pred = F.softmax(y_pred, dim=-1)
   # count how many predictions were correct, weighted for balance
   correct = torch.eq(y_pred.argmax(dim=1),labels).int()
   correct = torch.sum(weights * correct)

   return correct / nonzero