import torch
import logging
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


class BatchGenerator:
   def __init__(self,filelist,config_file):
      self.filelist     = np.array(filelist)
      self.evt_per_file = config_file['data_handling']['evt_per_file']
      self.batch_size   = config_file['training']['batch_size']
      self.img_shape    = config_file['data_handling']['image_shape']
      self.num_classes  = len(config_file['data_handling']['classes'])
      self.rank         = config_file['rank']
      self.nranks       = config_file['nranks']
      self.use_random   = config_file['data_handling']['shuffle']
      self.class_ids    = config_file['data_handling']['class_nums']
      self.eta_offset   = config_file['data_handling']['eta_offset']
      self.eta_diviser  = config_file['data_handling']['eta_diviser']
      self.phi_offset   = config_file['data_handling']['phi_offset']
      self.phi_diviser  = config_file['data_handling']['phi_diviser']
      self.r_offset     = config_file['data_handling']['r_offset']
      self.r_diviser    = config_file['data_handling']['r_diviser']

      self.total_images = self.evt_per_file * len(self.filelist)
      self.total_batches = self.total_images // self.batch_size // self.nranks
      self.batches_per_file = self.evt_per_file // self.batch_size

      self.files_per_rank = int(len(self.filelist) / self.nranks)

      self.running_class_count = np.zeros(self.num_classes)

   def set_random_batch_retrieval(self,flag=True):
      self.use_random = flag

   def __len__(self):
      return self.total_batches

   def batch_gen(self):
      if self.use_random:
         np.random.shuffle(self.filelist)

      start_file_index = self.rank * self.files_per_rank
      end_file_index = (self.rank + 1) * self.files_per_rank

      logger.warning('rank %s processing files %s through %s',self.rank,start_file_index,end_file_index)
      logger.warning('first file after shuffle: %s',self.filelist[0])
      image_counter = 0
      inputs = np.full([self.batch_size] + self.img_shape,0)
      targets = np.full([self.batch_size],0)

      for filename in self.filelist[start_file_index:end_file_index]:

         try:
            file = CSVFileGenerator(filename)
            input,target = file.get()
            inputs[image_counter,:input.shape[0],:input.shape[1]] = input
            targets[image_counter] = self.class_ids.index(target)
            self.running_class_count[int(targets[image_counter])] += 1
            image_counter += 1

            if image_counter == self.batch_size:
               inputs = self.normalize_inputs(inputs)
               inputs = torch.from_numpy(inputs).float().permute(0,2,1)
               targets = torch.from_numpy(targets).long()
               yield (inputs,targets)

               image_counter = 0
               inputs = np.zeros([self.batch_size] + self.img_shape)
               targets = np.zeros([self.batch_size])
         except ValueError:
            logger.exception('received exception while processing file %s',filename)

   def normalize_inputs(self,inputs):
      # shape: [B,N,4]
      inputs[...,0] = norm_mean_std(inputs[...,0])
      inputs[...,1] = norm_mean_std(inputs[...,1])
      inputs[...,2] = norm_mean_std(inputs[...,2])
      inputs[...,3] = norm_mean_std(inputs[...,3])

      return inputs


def norm_mean_std(x):
   mean = x.mean(axis=1).reshape(1,-1).transpose()
   std = x.std(axis=1).reshape(1,-1).transpose()
   return np.divide(x - mean,std,out=np.zeros_like(x,dtype='d'),where=std!=0)


class CSVFileGenerator:
   def __init__(self,filename):
      self.filename     = filename
      self.col_names    = ['id', 'index', 'x', 'y', 'z', 'eta', 'phi','r','Et','pid','true_pt']
      self.col_dtype    = {'id': np.int64, 'index': np.int64, 'x': np.float32, 'y': np.float32,
                           'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
                           'Et': np.float32, 'pid': np.int32, 'true_pt': np.float32}

   def open_file(self):
      try:
         logger.debug('opening file: %s',self.filename)
         self.data = pd.read_csv(self.filename,header=None,names=self.col_names, dtype=self.col_dtype, sep='\t')
      except:
         logger.exception('exception received when opening file %s',self.filename)
         raise

   def get(self):
      self.open_file()
      return (self.get_input(),self.get_target())

   def get_input(self):
      if hasattr(self,'data'):
         return self.data[['eta','phi','r','Et']]
      else:
         raise Exception('no data attribute')

   def get_target(self):
      if hasattr(self,'data'):
         return self.data['pid'][0]
      else:
         raise Exception('no data attribute')
