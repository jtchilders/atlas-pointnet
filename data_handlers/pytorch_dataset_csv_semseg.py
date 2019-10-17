from torch.utils import data as td
import torch
import logging,pandas as pd
import numpy as np
logger = logging.getLogger(__name__)


class CSVDataset(td.Dataset):
   def __init__(self,filelist,config):
      super(CSVDataset,self).__init__()
      self.config       = config
      batch_limiter = None
      if 'batch_limiter' in config:
         batch_limiter = config['batch_limiter']

      self.img_shape    = config['data_handling']['image_shape']
      self.class_ids    = config['data_handling']['class_nums']
      self.filelist     = filelist
      if batch_limiter is not None:
         logger.info('limiting filelist size to %s',batch_limiter)
         self.filelist = filelist[:batch_limiter]
      self.len          = len(self.filelist)
      # self.col_names    = ['id', 'index', 'x', 'y', 'z', 'eta', 'phi','r','Et','pid','true_pt']
      self.col_names    = ['id', 'index', 'x', 'y', 'z', 'r', 'eta', 'phi', 'Et','pid','n','trk_good','trk_id','trk_pt']
      # self.col_dtype    = {'id': np.int64, 'index': np.int64, 'x': np.float32, 'y': np.float32,
      #                      'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
      #                      'Et': np.float32, 'pid': np.int32, 'true_pt': np.float32}
      self.col_dtype    = {'id': np.int64, 'index': np.int32, 'x': np.float32, 'y': np.float32,
                           'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
                           'Et': np.float32, 'pid': np.float32, 'n': np.float32, 
                           'trk_good': np.float32, 'trk_id': np.float32, 'trk_pt': np.float32}

      self.class_map = {}
      for i,entry in enumerate(self.class_ids):
         self.class_map[entry] = i
         self.class_map[-entry] = i

   def __getitem__(self,index):
      filename = self.filelist[index]
      try:
         logger.debug('opening file: %s',filename)
         self.data = pd.read_csv(filename,header=None,names=self.col_names, dtype=self.col_dtype, sep='\t')
      except:
         logger.exception('exception received when opening file %s',filename)
         raise

      return self.get_input() + (self.get_target(),)

   def get_input(self):
      if hasattr(self,'data'):
         if 'silicon_only' in self.config['data_handling'] and self.config['data_handling']['silicon_only']:
            input = self.data[self.data['id'] < 4e17]
            input = input[['eta','phi','r','Et']]
         else:
            input = self.data[['eta','phi','r','Et']]

         input = input.to_numpy()
         input = np.float32(input)

         # create a weight vector with 1's where points exist and 0's where they do not
         weights = np.zeros(self.img_shape[0],dtype=np.float32)
         weights[0:input.shape[0]] = np.ones(input.shape[0],dtype=np.float32)

         padded_input = np.zeros(self.img_shape,dtype=np.float32)

         padded_input[:input.shape[0],:] = input

         input = padded_input.transpose()
         
         return input,weights
      else:
         raise Exception('no data attribute')

   def get_target(self):
      if hasattr(self,'data'):
         target = self.data['pid']
         # init_length = len(target)
         # logger.info('target[0:5] = %s',target.to_numpy()[0:5])
         # logger.info('map = %s',self.class_map)
         target = target.map(self.class_map)
         target = np.int32(target.to_numpy())

         padded_target = np.zeros(self.img_shape[0],dtype=np.int32)
         padded_target[:target.shape[0]] = target

         # logger.info('target[%s] = %s',init_length,target[init_length-5:init_length+5])
         return torch.from_numpy(padded_target)
      else:
         raise Exception('no data attribute')

   def __len__(self):
      return self.len

   @staticmethod
   def get_loader(dataset, batch_size=1,
                  shuffle=False, sampler=None, batch_sampler=None,
                  num_workers=0, pin_memory=False, drop_last=False,
                  timeout=0, worker_init_fn=None):

      return td.DataLoader(dataset,batch_size=batch_size,
                           shuffle=shuffle,sampler=sampler,batch_sampler=batch_sampler,
                           num_workers=num_workers,pin_memory=pin_memory,drop_last=drop_last,
                           timeout=timeout,worker_init_fn=worker_init_fn)


if __name__ == '__main__':
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging.basicConfig(level=logging.INFO,format=logging_format,datefmt=logging_datefmt)
   import sys,json,time
   import data_handlers.utils as du

   config = json.load(open(sys.argv[1]))

   tds,vds = du.get_datasets(config)

   for data in tds:
      image = data[0]
      label = data[1]

      logger.info('image = %s label = %s',image.shape,label)
      time.sleep(10)
