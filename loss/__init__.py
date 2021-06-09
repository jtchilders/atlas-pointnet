import torch
import logging
logger = logging.getLogger('losses')

__all__ = ['focal_loss']
from .focal_loss import focal_loss

def get_loss(config):
   loss_name = config['loss']['name']
   if loss_name in globals():
      logger.info('using loss name %s',loss_name)
      if 'args' in config['loss']:
         logging.info('passing args to loss function: %s',config['loss']['args'])
         return globals()[loss_name](**config['loss']['args'])
      else:
         return globals()[loss_name]
   elif hasattr(torch.nn.functional,loss_name):
      logger.info('using loss name %s',loss_name)
      return getattr(torch.nn.functional,loss_name)
   elif hasattr(torch.nn,loss_name):
      logger.info('using loss name %s',loss_name)
      if 'args' in config['loss']:
         logging.info('passing args to loss function: %s',config['loss']['args'])
         return getattr(torch.nn, loss_name)(**config['loss']['args'])
      else:
         return getattr(torch.nn,loss_name)()
   else:
      raise Exception('failed to find loss function %s' % loss_name)
