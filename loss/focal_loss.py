import torch

def focal_loss(labels,y_pred,gamma=2):
   """
   https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py
   Computer focal loss for multi classification
   Args:
      labels: A int32 tensor of shape [batch_size,points].
      logits: A float32 tensor of shape [batch_size,points,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
   Returns:
      A tensor of the same shape as `labels`
   """
   # y_pred = tf.nn.softmax(logits,dim=-1)  # [batch_size,points,num_classes]
   labels = tf.one_hot(labels,depth=y_pred.shape[2])  # [batch_size,points,num_classes]
   L = -labels * ((1 - y_pred) ** gamma) * torch.log(y_pred)  # [batch_size,points,num_classes]
   L = torch.sum(L,(1,2))  # [batch_size]
   return torch.mean(L)