import logging
logger = logging.getLogger('data_handlers')

__all__ = ['pytorch_dataset_csv_semseg']
from . import pytorch_dataset_csv_semseg

def get_datasets(config):

   if config['data']['handler'] in globals():
      logger.info('using data handler %s',config['data']['handler'])
      handler = globals()[config['data']['handler']]
   else:
      raise Exception('failed to find data handler %s in globals %s' % (config['data']['handler'],globals()))

   return handler.get_datasets(config)