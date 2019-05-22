import torch
import logging,multiprocessing
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

csv_mgr = None
csv_q = []


class BatchGeneratorPool(multiprocessing.Process):
   def __init__(self,filelist,config_file):
      super(BatchGeneratorPool,self).__init__()
      global csv_mgr,csv_q
      self.config_file  = config_file
      self.filelist     = np.array(filelist)
      self.evt_per_file = config_file['data_handling']['evt_per_file']
      self.pool_size    = config_file['data_handling']['pool_size']
      self.queue_depth  = config_file['data_handling']['queue_depth']
      self.batch_size   = config_file['training']['batch_size']
      self.img_shape    = config_file['data_handling']['image_shape']
      self.num_classes  = len(config_file['data_handling']['classes'])
      self.rank         = config_file['rank']
      self.nranks       = config_file['nranks']
      self.use_random   = config_file['data_handling']['shuffle']
      self.class_ids    = config_file['data_handling']['class_nums']

      self.total_images = self.evt_per_file * len(self.filelist)
      self.total_batches = self.total_images // self.batch_size // self.nranks
      self.batches_per_file = self.evt_per_file // self.batch_size

      self.files_per_rank = int(len(self.filelist) / self.nranks)

      self.running_class_count = np.zeros(self.num_classes)

      self.control_index = len(csv_q)
      if csv_mgr is None:
         csv_mgr = multiprocessing.Manager()
      csv_q.append(csv_mgr.Queue(maxsize=self.queue_depth))

   def __len__(self):
      return self.total_batches

   def batch_gen(self):

      while True:
         logger.debug('getting batch from queue')
         data = csv_q[self.control_index].get()
         logger.debug('queue get returned batch')
         if data:
            yield data
         else:
            break

   def run(self):

      # create a pool of workers with a subset of files
      logger.debug('creating pool size %s' % self.pool_size)
      p = multiprocessing.Pool(self.pool_size)

      # split batch indices
      batch_indices = np.arange(self.total_batches)
      batch_groups = np.array_split(batch_indices,self.pool_size)
      logger.debug('batch_groups = %s',len(batch_groups))
      inputs = tuple([(csv_q[self.control_index],self.batch_size,self.filelist,list(x),self.config_file) for x in batch_groups])
      logger.debug('inputs = %s',len(inputs))
      logger.debug('starting processes %s' % (len(batch_groups)))
      for batch_num,data in enumerate(p.imap_unordered(self.process_batch,inputs)):
         logger.debug('got batch %s with data sizes: %s',batch_num,data)

      logger.debug('all processes completed, closing pool')
      p.close()
      logger.debug('data reading done.')
      csv_q.put(None)

   @staticmethod
   def process_batch(inputs):
      logger.debug('in process_batch inputs = %s' % len(inputs))
      queue,batch_size,filelist,batch_indices,config_file = inputs
      logger.debug('in process_batch filelist = %s batch list = %s',len(filelist),len(batch_indices))
      for batch_index in batch_indices:
         logger.debug('in process_batch batch_index = %s' % batch_index)

         start_index = batch_size * batch_index
         end_index = batch_size * (batch_index + 1)
         logger.debug(f'in process_batch batch indices: {start_index} - {end_index}')
         batch_filelist = filelist[start_index:end_index]
         fllen = len(batch_filelist)
         logger.debug(f'in process_batch filelist: {batch_filelist} {batch_size}')

         bg = BatchGenerator(batch_filelist,config_file)

         logger.debug('in process_batch looping over batches %s',len(bg))
         for batch in bg.batch_gen():
            logger.debug('in process_batch putting batch on queue: %s',batch)
            queue.put(batch)
         logger.debug('in process_batch done looping over batches')

      return batch_indices


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

      self.total_images = self.evt_per_file * len(self.filelist)
      logger.debug('total images = %s evt_per_file = %s filelist = %s',self.total_images,self.evt_per_file,len(self.filelist))
      self.total_batches = self.total_images // self.batch_size // self.nranks
      self.batches_per_file = self.evt_per_file // self.batch_size

      self.files_per_rank = int(len(self.filelist) / self.nranks)

      self.running_class_count = np.zeros(self.num_classes)

   def set_random_batch_retrieval(self,flag=True):
      self.use_random = flag

   def __len__(self):
      return self.total_batches

   def batch_gen(self):
      if len(self.filelist) == 0:
         logger.error('filelist is empty')
         raise Exception('passed empty filelist')
      if self.use_random:
         np.random.shuffle(self.filelist)

      start_file_index = self.rank * self.files_per_rank
      end_file_index = (self.rank + 1) * self.files_per_rank
      logger.debug(f'filelist indices: {start_file_index} - {end_file_index}')

      logger.debug('rank %s processing files %s through %s',self.rank,start_file_index,end_file_index)
      logger.debug('first file after shuffle: %s',self.filelist[0])
      image_counter = 0
      inputs = np.zeros([self.batch_size] + self.img_shape)
      targets = np.zeros([self.batch_size])

      for filename in self.filelist[start_file_index:end_file_index]:

         try:
            file = CSVFileGenerator(filename)
            input,target = file.get()
            inputs[image_counter,...] = np.tile(input,(int(self.img_shape[0] / input.shape[0]) + 1,1))[:self.img_shape[0],...]
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


if __name__ == '__main__':
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   log_level = logging.DEBUG
   logging.basicConfig(level=log_level,format=logging_format,datefmt=logging_datefmt)
   import sys,json,glob,time
   print(sys.argv)
   config_file = json.load(open(sys.argv[1]))
   config_file['rank'] = 0
   config_file['nranks'] = 1

   filelist = glob.glob(config_file['data_handling']['train_glob'])
   if len(sys.argv) > 3 and len(filelist) > int(sys.argv[2]):
      filelist = filelist[:int(sys.argv[2])]
   logger.info('found %s files',len(filelist))

   bgp = BatchGeneratorPool(filelist,config_file)
   bgp.start()

   start = time.time()
   for data in bgp.batch_gen():
      end = time.time()
      logger.info('[%6.4f]got batch: %s',end-start,data)
      if data is None:
         break
      start = time.time()

   logger.info('done')


