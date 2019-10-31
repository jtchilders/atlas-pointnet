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

      # set target function
      self.get_target = getattr(self,config['data_handling']['target_func'])
      self.nothing_class = self.class_ids[config['data_handling']['nothing_class_index']]

      self.silicon_only = False
      if 'silicon_only' in self.config['data_handling'] and self.config['data_handling']['silicon_only']:
         self.silicon_only = True

      self.coords = ['eta','phi','r','Et']
      self.cartesian = False
      if 'coords' in self.config['data_handling']:
         if 'cartesian' in self.config['data_handling']['coords']:
            self.coords = ['x','y','z','Et']
            self.cartesian = True



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
      if self.silicon_only:
         input = self.data[self.data['id'] < 4e17]
         input = input[self.coords]
      else:
         input = self.data[self.coords]

      if not self.cartesian:
         input['r'] = input['r'] / 1000.
      else:
         input[['x','y','z']] /= 1000.
         input['Et'] /= 100.

      input = input.to_numpy()
      input = np.float32(input)

      # create a weight vector with 1's where points exist and 0's where they do not
      weights = np.zeros(self.img_shape[0],dtype=np.float32)
      weights[0:input.shape[0]] = np.ones(input.shape[0],dtype=np.float32)

      padded_input = np.zeros(self.img_shape,dtype=np.float32)

      padded_input[:input.shape[0],:] = input

      input = padded_input.transpose()
      
      return input,weights

   def class_crossentropy_targets(self):
      target = self.data['pid']

      target = target.map(self.class_map)
      target = np.int32(target.to_numpy())

      padded_target = np.zeros(self.img_shape[0],dtype=np.int32)
      padded_target[:target.shape[0]] = target

      return torch.from_numpy(padded_target)

   def bce_something_targets(self):
      target = np.int32(self.data['pid'].to_numpy())

      target = torch.from_numpy((target != self.nothing_class))

      padded_target = torch.zeros(self.img_shape[0],dtype=torch.int)
      padded_target[:target.shape[0]] = target

      return padded_target

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
