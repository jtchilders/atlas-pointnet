import torch
import logging
logger = logging.getLogger('optimizer')

__all__ = []

def get_optimizer(config):
   optimizer_name = config['optimizer']['name']
   if hasattr(torch.optim,optimizer_name):
      logger.info('using optimizer name %s',optimizer_name)
      return getattr(torch.optim,optimizer_name)
   else:
      raise Exception('failed to find optimizer function %s' % optimizer_name)


def get_learning_rate_scheduler(config):
   lrsched_name = config['lr_schedule']['name']
   if hasattr(torch.optim.lr_scheduler,lrsched_name):
      logger.info('using learning rate scheduler name %s',lrsched_name)
      return getattr(torch.optim.lr_scheduler,lrsched_name)
   else:
      raise Exception('failed to find learning rate scheduler function %s' % lrsched_name)

