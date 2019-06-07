from torch.utils import data as td
import torch
import h5py,logging
import numpy as np
from pytorch.model import device
logger = logging.getLogger(__name__)

class ImageDataset(td.Dataset):
   def __init__(self,filelist,config):
      super(ImageDataset,self).__init__()
      self.config = config
      self.image_shape = config['data_handling']['image_shape']
      self.filelist = filelist
      self.grid_size = None
      self.images_per_file = config['data_handling']['images_per_file']
      self.len = len(self.filelist) * self.images_per_file

      self.last_file_index = -1
      self.last_file = None

   def __getitem__(self,index):
      # logger.info('getting index %s',index)
      image_index = self.get_image_index(index)
      file_index = self.get_file_index(index)

      if file_index == self.last_file_index:
         file = self.last_file
      else:
         self.last_file_index = file_index
         self.last_file = h5py.File(self.filelist[file_index],'r')
         file = self.last_file

      image = np.float32(file['raw'][image_index])
      truth = np.int32(file['truth'][image_index])
      truth = self.convert_truth_classonly(truth,self.image_shape[1],self.image_shape[2])
      
      image = torch.from_numpy(image)
      truth = torch.from_numpy(truth)

      image = image.to(device)
      truth = truth.to(device)

      return image,truth

   def convert_truth_classonly(self,truth,img_height,img_width):

      pix_per_grid_h = img_height / self.grid_size[0]
      pix_per_grid_w = img_width / self.grid_size[1]

      new_truth = np.zeros((2,self.grid_size[0],self.grid_size[1]),dtype=np.int32)

      for obj_num in range(len(truth)):
         obj_truth = truth[obj_num]

         obj_exists   = obj_truth[0]

         if obj_exists == 1:

            obj_center_x = obj_truth[1] / pix_per_grid_w
            obj_center_y = obj_truth[2] / pix_per_grid_h
            # obj_width    = obj_truth[3] / pix_per_grid_w
            # obj_height   = obj_truth[4] / pix_per_grid_h

            grid_x = int(np.floor(obj_center_x))
            grid_y = int(np.floor(obj_center_y))

            # if grid_x >= self.grid_w:
            #    raise Exception('grid_x %s is not less than grid_w %s' % (grid_x,self.grid_w))
            # if grid_y >= self.grid_h:
            #    raise Exception('grid_y %s is not less than grid_h %s' % (grid_y,self.grid_h))

            new_truth[0,grid_y,grid_x] = obj_exists
            new_truth[1,grid_y,grid_x] = np.argmax([np.sum(obj_truth[5:10]),np.sum(obj_truth[10:12])])

      return new_truth

   def get_image_index(self,index):
      return index % self.images_per_file

   def get_file_index(self,index):
      return index // self.images_per_file

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
   tds.dataset.grid_size = (10,10)  # these come from the model
   vds.dataset.grid_size = (10,10)  # these come frmo the model

   for data in tds:
      image = data[0]
      label = data[1]

      logger.info('image = %s label = %s',image.shape,label.shape)
      time.sleep(10)


