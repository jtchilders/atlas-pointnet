import logging,glob
logger = logging.getLogger(__name__)



def get_filelist(config_file):
   # get file list
   logger.info('train glob dir: %s',config_file['data_handling']['train_glob'])
   logger.info('valid glob dir: %s',config_file['data_handling']['valid_glob'])
   train_filelist = sorted(glob.glob(config_file['data_handling']['train_glob']))
   valid_filelist = sorted(glob.glob(config_file['data_handling']['valid_glob']))
   logger.info('found %s training files, %s validation files',len(train_filelist),len(valid_filelist))

   if len(train_filelist) < 1 or len(valid_filelist) < 1:
      raise Exception('length of file list needs to be at least 1 for train (%s) and val (%s) samples',len(train_filelist),len(valid_filelist))

   logger.warning('first file: %s',train_filelist[0])

   return train_filelist,valid_filelist
