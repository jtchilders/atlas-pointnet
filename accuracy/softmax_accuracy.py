import torch

def softmax_accuracy(y_pred,labels,weights):
   # number of non-zero points in this batch
   nonzero = torch.sum(weights)

   # count how many predictions were correct, weighted for balance
   correct = torch.sum(weights * torch.eq(y_pred,labels).type(torch.int32))

   return correct / nonzero