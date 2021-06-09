import torch
import logging
logger = logging.getLogger('accuracy')

__all__ = ['softmax_accuracy']
from .softmax_accuracy import softmax_accuracy

def get_accuracy(config):
   acc_name = config['accuracy']['name']
   if acc_name in globals():
      logger.info('using accuracy name %s',acc_name)
      if 'args' in config['accuracy']:
         logging.info('passing args to accuracy function: %s',config['accuracy']['args'])
         return globals()[acc_name](**config['accuracy']['args'])
      else:
         return globals()[acc_name]
   else:
      raise Exception('failed to find accuracy function %s' % acc_name)