import torch
import logging
import pandas as pd
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger(__name__)


labels_dict =  {0:0,1:0, -1:0, 2:0, -2:0, 3:0, -3:0, 4:0, -4:0, 
                5:0, -5:0, 11:1,-11:1,13:2,-13:2,-99:2, 15:2, -15: 2}

col_names    = ['id', 'index', 'x', 'y', 'z', 'r', 'eta', 'phi',
                'Et','pid','pn','peta','pphi','ppt','trk_good','trk_id','trk_pt']
col_dtype    = {'id': np.int64, 'index': np.int32, 'x': np.float32, 'y': np.float32,
                'z': np.float32, 'eta': np.float32, 'phi': np.float32, 'r': np.float32,
                'Et': np.float32, 'pid': np.int32, 'pn': np.int32, 'peta': np.float32,
                'pphi': np.float32, 'ppt': np.float32,
                'trk_good': np.float32, 'trk_id': np.float32, 'trk_pt': np.float32}
include_cols = ['x','y','z','eta','phi','r','Et']


def normalize(tensor):
   tmin = tensor.min()
   tmax = tensor.max()
   dist = tmax - tmin
   tensor = (tensor - tmin) / dist
   return tensor


def get_datasets(config):
   global gconfig,gnum_points,gnum_features

   train = CSVDataset.from_filelist(config['data']['train_filelist'],config,True)
   valid = CSVDataset.from_filelist(config['data']['test_filelist'],config,False)
   
   return train,valid

class CSVDataset(torch.utils.data.Dataset):
   def __init__(self,filelist,config,training):
      super(CSVDataset,self).__init__()
      self.config        = config
      self.num_points    = config['data']['num_points']
      self.num_features  = config['data']['num_features']
      self.num_classes   = config['data']['num_classes']
      self.data_rotation = config['data']['rotation']
      self.xyz_norm      = config['data']['xyz_norm']
      self.filelist      = filelist
      self.training      = training
      if hasattr(np,config['data']['dtype']):
         self.dtype      = getattr(np,config['data']['dtype'])


   def __len__(self):
      return len(self.filelist)

   def __getitem__(self,index):
      filename = self.filelist[index]
      try:
         logger.debug('opening file: %s',filename)
         df = pd.read_csv(filename,header=None,names=col_names, dtype=col_dtype, sep='\t')
      except:
         logger.exception('exception received when opening file %s',filename)
         raise

      return self.build_data(df)

   def build_data(self,df):
      
      # clip the number of points from the input based on the config num_points
      if len(df) > self.num_points:
         df = df[0:self.num_points]

      if self.data_rotation and self.training:
         rotation_angle,rotation_matrix = self.random_rotation()
         # logger.info('old (x,y,z) = %s  old eta,phi = %s',df[['x','y','z']].to_numpy()[0],df[['eta','phi']].to_numpy()[0])
         df[['x','y','z']] = np.dot(df[['x','y','z']],rotation_matrix)
         df['phi'] = df['phi'] + rotation_angle  # phi is -pi to pi, rotation angle is 0 - 2pi
         df['phi'] = df['phi'].apply(lambda x: x if x < np.pi else x - 2 * np.pi)
         # logger.info('new (x,y,z) = %s  new eta,phi = %s',df[['x','y','z']].to_numpy()[0],df[['eta','phi']].to_numpy()[0])

      if self.xyz_norm:
         df['x'] = normalize(df['x'])
         df['y'] = normalize(df['y'])
         df['z'] = normalize(df['z'])
         df['r'] = normalize(df['r'])

      jagged_inputs = df[include_cols].to_numpy()


      # stuff ragged event sizes into fixed size
      inputs = np.zeros([self.num_points,self.num_features],dtype=self.dtype)
      # logger.info('3 inputs: %s',inputs.shape)
      inputs[0:jagged_inputs.shape[0],...] = jagged_inputs[0:jagged_inputs.shape[0],...]

      # build the labels
      jagged_labels = df.pid
      # map pid to class label
      jagged_labels = jagged_labels.map(labels_dict)
      # convert to numpy
      jagged_labels = jagged_labels.to_numpy()

      # count number of each class
      # use the lowest to decide weights for loss function
      # get list of unique classes and their occurance count
      unique_classes,unique_counts = np.unique(jagged_labels,return_counts=True)
      # logger.info('unique_classes = %s',unique_classes)
      # logger.info('unique_counts = %s',unique_counts)
      # get mininum class occurance count
      min_class_count = np.min(unique_counts)
      # logger.info('min_class_count = %s',min_class_count)
      # create class weights to be applied to loss as mask
      # this will balance the loss function across classes
      class_weights = np.zeros([self.num_points],dtype=np.int32)
      # logger.info('class_weights.shape = %s',class_weights.shape)
      # set weights to one for an equal number of classes
      for class_label in unique_classes:
         # logger.info('class_label = %s',class_label)
         class_indices = np.nonzero(jagged_labels == class_label)[0]
         # logger.info('class_indices.shape = %s',class_indices.shape)
         class_indices = np.random.choice(class_indices,size=[min_class_count],replace=False)
         # logger.info('class_indices.shape = %s',class_indices.shape)
         class_weights[class_indices] = 1
         # logger.info('class_weights.sum = %s',class_weights.sum())
         # logger.info('labels_weights.sum = %s',df_labels[class_indices].sum())
      # logger.info('class_weights unique = %s',np.unique(class_weights,return_counts=True))

      nonzero_mask = np.zeros([self.num_points],dtype=np.int32)
      nonzero_mask[0:jagged_labels.shape[0]] = 1

      # pad with zeros or clip some points
      labels = np.zeros([self.num_points])
      labels[0:jagged_labels.shape[0]] = jagged_labels[0:jagged_labels.shape[0]]

      return inputs,labels,class_weights,nonzero_mask

   @staticmethod
   def random_rotation():
      rotation_angle = np.random.uniform() * 2 * np.pi
      cosval = np.cos(rotation_angle)
      sinval = np.sin(rotation_angle)
      rotation_matrix = np.array([[cosval, -sinval, 0],
                                 [sinval, cosval, 0],
                                 [0, 0, 1]])
      
      return rotation_angle,rotation_matrix

   @staticmethod
   def from_filelist(filelist_filename,config,training):
      filelist = []
      basedir = ''
      if 'filebase' in config['data']:
         basedir = config['data']['filebase'] + '/'
      for line in open(filelist_filename):
         filelist.append(basedir + line.strip())

      return CSVDataset(filelist,config,training)


if __name__ == '__main__':
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging.basicConfig(level=logging.INFO,format=logging_format,datefmt=logging_datefmt)
   import sys,json,time

   config = json.load(open(sys.argv[1]))

   trainds,testds = get_datasets(config)

   ## create samplers for these datasets
   train_sampler = torch.utils.data.distributed.DistributedSampler(trainds,1,0,shuffle=True,drop_last=True)
   test_sampler = torch.utils.data.distributed.DistributedSampler(testds,1,0,shuffle=True,drop_last=True)

   ## create data loaders
   batch_size = config['data']['batch_size']
   train_loader = torch.utils.data.DataLoader(trainds,shuffle=False,
                                            sampler=train_sampler,num_workers=config['data']['num_parallel_readers'],
                                            batch_size=batch_size,persistent_workers=True)
   test_loader = torch.utils.data.DataLoader(testds,shuffle=False,
                                            sampler=test_sampler,num_workers=config['data']['num_parallel_readers'],
                                            batch_size=batch_size,persistent_workers=True)

   for batch_number,(inputs,labels,class_weights,nonzero_mask) in enumerate(train_loader):

      unique_classes,unique_counts = np.unique(labels*class_weights,return_counts=True)
      
      logger.info('batch_number = %s input shape = %s    labels shape = %s',batch_number,inputs.shape,labels.shape)
      #logger.info('batch_number = %s labels = %s',batch_number,np.squeeze(labels[0:10].numpy()).tolist())
      logger.info(f'unique_classes={unique_classes} unique_counts={unique_counts}')
      time.sleep(10)
