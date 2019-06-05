import logging,glob
logger = logging.getLogger(__name__)


def get_filelist(config_file):
   # get file list
   batch_limiter = None
   if 'batch_limiter' in config_file:
      batch_limiter = config_file['batch_limiter']
   logger.info('train glob dir: %s',config_file['data_handling']['train_glob'])
   logger.info('valid glob dir: %s',config_file['data_handling']['valid_glob'])
   train_filelist = sorted(glob.glob(config_file['data_handling']['train_glob']))
   if batch_limiter:
      maxfile = batch_limiter * config_file['training']['batch_size'] / config_file['data_handling']['evt_per_file']
      train_filelist = train_filelist[0:int(maxfile + 1)]
   valid_filelist = sorted(glob.glob(config_file['data_handling']['valid_glob']))
   logger.info('found %s training files, %s validation files',len(train_filelist),len(valid_filelist))

   if len(train_filelist) < 1 or len(valid_filelist) < 1:
      raise Exception('length of file list needs to be at least 1 for train (%s) and val (%s) samples',len(train_filelist),len(valid_filelist))

   logger.warning('first file: %s',train_filelist[0])

   return train_filelist,valid_filelist


def get_datasets(config_file):

   logger.info('getting filelists')
   trainlist,validlist = get_filelist(config_file)

   if 'csv' == config_file['data_handling']['input_format']:
      logger.info('using CSV data handler')
      from data_handlers.csv_format import BatchGenerator
      logger.info('creating batch generators')
      trainds = BatchGenerator(trainlist,config_file,'BatchGen:train')
      validds = BatchGenerator(validlist,config_file,'BatchGen:valid')

   elif 'dataset_csv' == config_file['data_handling']['input_format']:
      logger.info('using CSV Dataset data handler')
      from data_handlers.pytorch_dataset_csv import CSVDataset
      logger.info('creating batch generators')
      traindss = CSVDataset(trainlist,config_file)
      trainds = CSVDataset.get_loader(traindss,batch_size=config_file['training']['batch_size'],
                                      shuffle=config_file['data_handling']['shuffle'],
                                      num_workers=config_file['data_handling']['workers'])
      validdss = CSVDataset(validlist,config_file)
      validds = CSVDataset.get_loader(validdss,batch_size=config_file['training']['batch_size'],
                                      shuffle=config_file['data_handling']['shuffle'],
                                      num_workers=config_file['data_handling']['workers'])

   elif 'csv_pool' == config_file['data_handling']['input_format']:
      logger.info('using CSV pool data handler')
      from data_handlers.csv_format import BatchGeneratorPool
      logger.info('creating batch generators')
      trainds = BatchGeneratorPool(trainlist,config_file,'BatchGenPool:train')
      trainds.start()
      validds = BatchGeneratorPool(validlist,config_file,'BatchGenPool:valid')
      validds.start()
   elif 'dataset_h5' == config_file['data_handling']['input_format']:
      logger.info('using H5 Dataset')
      from data_handlers.pytorch_dataset_h5 import ImageDataset
      traindss = ImageDataset(trainlist,config_file)
      trainds = ImageDataset.get_loader(traindss,batch_size=config_file['training']['batch_size'],
                                        shuffle=config_file['data_handling']['shuffle'],
                                        num_workers=config_file['data_handling']['workers'])
      validdss = ImageDataset(validlist,config_file)
      validds = ImageDataset.get_loader(validdss,batch_size=config_file['training']['batch_size'],
                                        shuffle=config_file['data_handling']['shuffle'],
                                        num_workers=config_file['data_handling']['workers'])
   else:
      raise Exception('no input file format specified in configuration')

   return trainds,validds
