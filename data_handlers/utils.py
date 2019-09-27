import logging,glob,json
import numpy as np
logger = logging.getLogger(__name__)


def get_filelist(config_file):
   if config_file['valid_only'] and 'valid_json' in config_file['data_handling']:
      return get_filelistC(config_file)
   elif 'train_glob' in config_file['data_handling'] and 'valid_glob' in config_file['data_handling']:
      return get_filelistA(config_file)
   elif 'glob' in config_file['data_handling']:
      return get_filelistB(config_file)
   else:
      raise Exception('must define ["glob"] OR ["train_glob" AND "valid_glob"] OR ["valid_json" AND --valid_only] in the "data_handing" section of json config file')


def get_filelistA(config_file):
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

   #logger.warning('first file: %s',train_filelist[0])

   return train_filelist,valid_filelist


def get_filelistB(config_file):
   # batch_limiter = None
   # if 'batch_limiter' in config_file:
   #    batch_limiter = config_file['batch_limiter']

   # get glob string
   glob_str = config_file['data_handling']['glob']
   logger.info('using glob string to find input files: %s',glob_str)
   batch_size = config_file['training']['batch_size']

   # glob for full filelist
   full_filelist = sorted(glob.glob(glob_str))
   # randomly shuffle the list a few times
   np.random.shuffle(full_filelist)
   np.random.shuffle(full_filelist)
   np.random.shuffle(full_filelist)
   np.random.shuffle(full_filelist)

   nfiles = len(full_filelist)

   if 'train_fraction' not in config_file['data_handling']:
      raise Exception('when using the "glob" setting, you must also specify the fraction of the total files to use for training by setting the "train_fraction" setting, the remaining will be used for validation.')

   # get training fraction
   trainfrac = config_file['data_handling']['train_fraction']
   ntrain = int(nfiles * trainfrac)

   train_filelist = full_filelist[:ntrain]
   
   # calculate number of files to use for training
   # train_filelist = train_filelist[:(len(train_filelist) // batch_size) * batch_size]

   # if batch_limiter:
   #    maxfile = batch_limiter * config_file['training']['batch_size'] / config_file['data_handling']['evt_per_file']
   #    train_filelist = train_filelist[0:int(maxfile + 1)]

   valid_filelist = full_filelist[ntrain:]
   # valid_filelist = valid_filelist[:(len(valid_filelist) // batch_size) * batch_size]
   logger.info('found %s training files, %s validation files',len(train_filelist),len(valid_filelist))
   
   if len(train_filelist) < 1 or len(valid_filelist) < 1:
      raise Exception('length of file list needs to be at least 1 for train (%s) and val (%s) samples',len(train_filelist),len(valid_filelist))

   #logger.warning('first file: %s',train_filelist[0])

   json.dump(train_filelist,open(config_file['filelist_base'] + '.trainlist','w'),indent=4, sort_keys=True)
   json.dump(valid_filelist,open(config_file['filelist_base'] + '.validlist','w'),indent=4, sort_keys=True)
   
   return train_filelist,valid_filelist


def get_filelistC(config_file):
   list = json.load(open(config_file['data_handling']['valid_json']))
   return list,list


def get_shard(config_file,filelist):
   rank = config_file['rank']
   nranks = config_file['nranks']

   # calc files per shard
   files_per_rank = len(filelist) // nranks
   # starting file
   start = rank * files_per_rank
   end   = (rank + 1) * files_per_rank

   return filelist[start:end]


def get_datasets(config_file):

   logger.info('getting filelists')
   trainlist,validlist = get_filelist(config_file)

   hvd = config_file['hvd']
   rank = config_file['rank']
   nranks = config_file['nranks']

   if 'csv' == config_file['data_handling']['input_format']:
      logger.info('using CSV data handler')
      from data_handlers.csv_format import BatchGenerator
      logger.info('creating batch generators')
      trainds = BatchGenerator(trainlist,config_file,'BatchGen:train')
      validds = BatchGenerator(validlist,config_file,'BatchGen:valid')
   elif 'dataset_csv_semseg' == config_file['data_handling']['input_format']:
      logger.info('using CSV Dataset data handler for semantic segmentation')
      from data_handlers.pytorch_dataset_csv_semseg import CSVDataset
      logger.info('creating batch generators')
      traindss = CSVDataset(trainlist,config_file)
      train_sampler = None
      train_shuffle = config_file['data_handling']['shuffle']
      if hvd is not None:
         import torch
         train_sampler = torch.utils.data.distributed.DistributedSampler(traindss,num_replicas=nranks,rank=rank)
         train_shuffle = False
      trainds = CSVDataset.get_loader(traindss,batch_size=config_file['training']['batch_size'],
                                      shuffle=train_shuffle,
                                      num_workers=config_file['data_handling']['workers'],
                                      sampler=train_sampler)
      validdss = CSVDataset(validlist,config_file)
      validds = CSVDataset.get_loader(validdss,batch_size=config_file['training']['batch_size'],
                                      shuffle=config_file['data_handling']['shuffle'],
                                      num_workers=0)
   elif 'dataset_csv' == config_file['data_handling']['input_format']:
      logger.info('using CSV Dataset data handler')
      from data_handlers.pytorch_dataset_csv import CSVDataset
      logger.info('creating batch generators')
      traindss = CSVDataset(trainlist,config_file)
      train_sampler = None
      train_shuffle = config_file['data_handling']['shuffle']
      if hvd is not None:
         import torch
         train_sampler = torch.utils.data.distributed.DistributedSampler(traindss,num_replicas=nranks,rank=rank)
         train_shuffle = False
      trainds = CSVDataset.get_loader(traindss,batch_size=config_file['training']['batch_size'],
                                      shuffle=train_shuffle,
                                      num_workers=config_file['data_handling']['workers'],
                                      sampler=train_sampler)
      validdss = CSVDataset(validlist,config_file)
      validds = CSVDataset.get_loader(validdss,batch_size=config_file['training']['batch_size'],
                                      shuffle=config_file['data_handling']['shuffle'],
                                      num_workers=0)

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
      raise Exception('no input file format specified in configuration, setting is %s' % config_file['data_handling']['input_format'])

   logger.info('generated training and validation datasets: %s %s',len(trainds),len(validds))

   return trainds,validds
