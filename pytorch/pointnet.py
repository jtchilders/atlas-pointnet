import torch
import pytorch.utils as utils
import logging,time
from sklearn.metrics import confusion_matrix
import CalcMean
import numpy as np
from pytorch.model import device
logger = logging.getLogger(__name__)


class PointNet2d(torch.nn.Module):
   def __init__(self,nChannels,nPoints,nCoords,nClasses):
      super(PointNet2d,self).__init__()

      self.input_trans = Transform2d(nPoints,nCoords,nChannels,3,initial_reshape=True)
      
      self.conv64A = utils.Conv2d(nChannels,64,(1,nCoords),pool=False)
      self.conv64B = utils.Conv2d(64,64,(1,1),pool=False)
      
      self.feature_trans = Transform2d(nPoints,1,64,64)
      
      self.conv64C = utils.Conv2d(64,64,(1,1),pool=False)
      self.conv128  = utils.Conv2d(64,128,(1,1),pool=False)
      self.conv1024 = utils.Conv2d(128,1024,(1,1),pool=False)
      
      self.pool = torch.nn.MaxPool2d((nPoints,1))
      
      self.linear512 = utils.Linear(1024,512)
      self.dropoutA = torch.nn.Dropout(0.7)
      
      self.linear256 = utils.Linear(512,256)
      self.dropoutB = torch.nn.Dropout(0.7)
   
      self.linearID = utils.Linear(256,nClasses,bn=False,activation=None)
      
   def forward(self,x):
      batch_size = x.shape[0]
      
      it = self.input_trans(x)
      endpoints = {'input_trans':it}
      x = torch.matmul(x,it)
      
      x = x.reshape((batch_size,1) + x.shape[1:])
      
      x = self.conv64A(x)
      x = self.conv64B(x)
      
      ft = self.feature_trans(x)
      endpoints['feature_trans'] = ft
      x = torch.matmul(ft,x.squeeze(dim=-1))
      
      x = x.reshape((batch_size,) + x.shape[1:] + (1,))
      
      x = self.conv64C(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      
      x = self.linear512(x)
      x = self.dropoutA(x)
      
      x = self.linear256(x)
      x = self.dropoutB(x)
      
      x = self.linearID(x)
      
      return x,endpoints


class Transform2d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width,channels,K,initial_reshape=False):
      super(Transform2d,self).__init__()

      self.K = K
      self.initial_reshape = initial_reshape

      self.conv64 = utils.Conv2d(channels,64,(1,width),pool=False)
      self.conv128 = utils.Conv2d(64,128,(1,1),pool=False)
      self.conv1024 = utils.Conv2d(128,1024,(1,1),pool=False)
      
      self.pool = torch.nn.MaxPool2d((height,1))
      
      self.linear512 = utils.Linear(1024,512)
      self.linear256 = utils.Linear(512,256)

      self.weights = torch.zeros(256,K * K,requires_grad=True)
      self.biases  = torch.eye(K,requires_grad=True).flatten()

   def forward(self,x):
      batch_size = x.shape[0]
      
      if self.initial_reshape:
         x = x.reshape((batch_size,1) + x.shape[1:])
      
      x = self.conv64(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = torch.reshape(x,[batch_size,-1])
      x = self.linear512(x)
      x = self.linear256(x)
      
      x = torch.matmul(x,self.weights) + self.biases
      x = torch.reshape(x,[batch_size,self.K,self.K])
      
      return x


class PointNet1d(torch.nn.Module):
   def __init__(self,config):
      super(PointNet1d,self).__init__()

      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 2)

      nPoints = input_shape[0]
      nCoords = input_shape[1]
      nClasses = len(config['data_handling']['classes'])
      
      logger.debug('nPoints = %s, nCoords = %s, nClasses = %s',nPoints,nCoords,nClasses)

      self.input_trans = Transform1d(nPoints,nCoords)
      
      self.input_to_feature = torch.nn.Sequential()
      for x in config['model']['input_to_feature']:
         N_in,N_out,pool = x
         self.input_to_feature.add_module(f'conv_{N_in}_to_{N_out}',utils.Conv1d(N_in,N_out,pool=pool))
         #utils.Conv1d(nCoords,64,pool=False)
         #utils.Conv1d(64,64,pool=False))
      
      self.feature_trans = Transform1d(nPoints,config['model']['input_to_feature'][-1][1])
      
      self.feature_to_pool = torch.nn.Sequential()
      for x in config['model']['feature_to_pool']:
         N_in,N_out,pool = x
         self.feature_to_pool.add_module(f'conv_{N_in}_to_{N_out}',utils.Conv1d(N_in,N_out,pool=pool))
         
      self.pool = torch.nn.MaxPool1d(nPoints)

      self.dense_layers = torch.nn.Sequential()
      for x in config['model']['dense_layers']:
         N_in,N_out,dropout,bn,act = x
         dr = int(dropout * 3)
         self.dense_layers.add_module(f'dense_{N_in}_to_{N_out}',utils.Linear(N_in,N_out,bn=bn,activation=act))
         if dropout > 0:
            self.dense_layers.add_module(f'dropout_{dr:03d}',torch.nn.Dropout(dropout))
      
   def forward(self,x):
      batch_size = x.shape[0]
      # l=10
      # print('pointnet1d input: %s' % x[0,:l,:])
      it = self.input_trans(x)
      # print('pointnet1d input_trans: %s' % it[0])
      endpoints = {'input_trans':it}
      x = torch.bmm(it,x)
      # print('pointnet1d it*x: %s' % x[0,:l,:l])
      
      x = self.input_to_feature(x)
      
      ft = self.feature_trans(x)
      # print('pointnet1d feature_trans: %s' % ft[0,:10,:10])
      endpoints['feature_trans'] = ft
      x = torch.bmm(ft,x)
      # print('pointnet1d ft*x: %s' % x[0,:l,:l])
      
      x = self.feature_to_pool(x)
      
      x = self.pool(x)
      # print('pointnet1d pool: %s' % x[0,:l,:])
      
      x = x.reshape([batch_size,-1])
      # print('pointnet1d reshape: %s' % x[0,:l])
      
      x = self.dense_layers(x)
      
      return x,endpoints

   def train_model(self,opt,lrsched,trainds,validds,config,writer=None):

      batch_size = config['training']['batch_size']
      status = config['status']
      epochs = config['training']['epochs']
      nval = config['nval']
      nval_tests = config['nval_tests']
      nsave = config['nsave']
      model_save = config['model_save']

      # some data handlers need a restart
      if callable(getattr(validds,'start_epoch',None)):
         validds.start_epoch()

      # get data iterator for validation
      if callable(getattr(validds,'batch_gen',None)):
         logger.info('using batch_gen method for valid')
         validds_itr = iter(validds.batch_gen())
      else:
         validds_itr = validds

      loss = self.get_loss(config)
      acc = self.get_accuracy(config)

      data_time = CalcMean.CalcMean()
      move_time = CalcMean.CalcMean()
      forward_time = CalcMean.CalcMean()
      backward_time = CalcMean.CalcMean()
      batch_time = CalcMean.CalcMean()
      loss_time = CalcMean.CalcMean()
      acc_time = CalcMean.CalcMean()

      monitor_loss = CalcMean.CalcMean()

      valid_batch_counter = 0
      for epoch in range(epochs):
         logger.info(' epoch %s of %s',epoch,epochs)

         # some data handlers need a restart
         if callable(getattr(trainds,'start_epoch',None)):
            trainds.start_epoch()

         # get data iterator
         if callable(getattr(trainds,'batch_gen',None)):
            logger.info('using batch_gen method for train')
            trainds_itr = iter(trainds.batch_gen())
         else:
            trainds_itr = trainds

         if lrsched:
            lrsched.step()

         for param_group in opt.param_groups:
            logging.info('learning rate: %s',param_group['lr'])
            if writer:
               writer.add_scalar('learning_rate',param_group['lr'],epoch)

         self.train()
         batch_counter = 0
         start_data = time.time()
         for batch_data in trainds_itr:
            end_data = time.time()
            
            # logger.debug('got training batch %s',batch_counter)
            start_move = end_data
            inputs = batch_data[0]
            inputs = inputs.to(device)
            targets = batch_data[1]
            targets = targets.to(device)
            end_move = time.time()

            # logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

            opt.zero_grad()
            # logger.debug('zeroed opt')
            start_forward = time.time()
            outputs,endpoints = self(inputs)
            end_forward = time.time()
            # logger.debug('got outputs: %s targets: %s',outputs,targets)

            start_loss = end_forward
            loss_value = loss(outputs,targets,endpoints,device=device)
            end_loss = time.time()
            monitor_loss.add_value(loss_value)
            # logger.debug('got loss')

            start_acc = time.time()
            acc_value = acc(outputs,targets)
            end_acc = time.time()

            start_backward = end_acc
            loss_value.backward()
            opt.step()

            end_backward = time.time()

            data_time.add_value(end_data - start_data)
            move_time.add_value(end_move - start_move)
            forward_time.add_value(end_forward - start_forward)
            backward_time.add_value(end_backward - start_backward)
            batch_time.add_value(end_backward - start_data)
            loss_time.add_value(end_loss - start_loss)
            acc_time.add_value(end_acc - start_acc)

            batch_counter += 1

            del inputs,targets

            # print statistics
            if config['rank'] == 0:
               if batch_counter % status == 0:
                  mean_img_per_second = (forward_time.calc_mean() + backward_time.calc_mean()) / batch_size
                  mean_img_per_second = 1. / mean_img_per_second
                  
                  logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f train acc: %6.4f  images/sec: %6.2f   data time: %6.3f move time: %6.3f forward time: %6.3f loss time: %6.3f  backward time: %6.3f acc time: %6.3f inclusive time: %6.3f',epoch + 1,epochs,batch_counter,len(trainds),monitor_loss.calc_mean(),acc_value.item(),mean_img_per_second,data_time.calc_mean(),move_time.calc_mean(),forward_time.calc_mean(),acc_time.calc_mean(),backward_time.calc_mean(),acc_time.calc_mean(),batch_time.calc_mean())
                  # logger.info('running count = %s',trainds.running_class_count)
                  # logger.info('prediction = %s',torch.nn.Softmax(dim=1)(outputs))

                  if writer:
                     global_batch = epoch * len(trainds) + batch_counter
                     writer.add_scalars('loss',{'train':monitor_loss.calc_mean()},global_batch)
                     writer.add_scalars('accuracy',{'train':acc_value.item()},global_batch)
                     writer.add_scalar('image_per_second',mean_img_per_second,global_batch)

                  monitor_loss = CalcMean.CalcMean()

               if batch_counter % nval == 0 or batch_counter == len(trainds):
                  #logger.info('running validation')
                  self.eval()

                  # if validds.reached_end:
                  #    logger.warning('restarting validation file pool.')
                  #    validds.start_file_pool(1)

                  vloss = CalcMean.CalcMean()
                  vacc = CalcMean.CalcMean()

                  valid_batch_counter = 0
                  for batch_data in validds_itr:

                     valid_batch_counter += 1
                     
                     inputs = batch_data[0].to(device)
                     targets = batch_data[1].to(device)

                     # logger.debug('valid inputs: %s targets: %s',inputs.shape,targets.shape)

                     outputs,endpoints = self(inputs)
                     #true_positive_accuracy,true_or_false_positive_accuracy,filled_grids_accuracy = accuracyCalc.eval_acc(outputs,targets,inputs)

                     loss_value = loss(outputs,targets,endpoints,device=device)
                     vloss.add_value(loss_value.item())
                     acc_value = acc(outputs,targets)
                     vacc.add_value(acc_value.item())

                     if writer:
                        global_batch = epoch * len(trainds) + batch_counter
                        writer.add_scalars('loss',{'valid':loss_value.item()},global_batch)
                        writer.add_scalars('accuracy',{'valid':acc_value.item()},global_batch)

                     if valid_batch_counter > nval_tests:
                        break
                     
                  logger.info('>[%3d of %3d, %5d of %5d]<<< valid loss: %6.4f valid acc: %6.4f >>>',epoch + 1,epochs,batch_counter,len(trainds),vloss.calc_mean(),vacc.calc_mean())

                  self.train()

               if batch_counter % nsave == 0:
                  torch.save(self.state_dict(),model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

            start_data = time.time()

   def valid_model(self,validds,config):

      batch_size = config['training']['batch_size']
      status = config['status']
      nClasses = len(config['data_handling']['classes'])

      data_time = CalcMean.CalcMean()
      forward_time = CalcMean.CalcMean()
      backward_time = CalcMean.CalcMean()
      batch_time = CalcMean.CalcMean()

      self.eval()
      self.to(device)
      batch_counter = 0
      start_data = time.time()

      confmat = np.zeros((nClasses,nClasses))
      
      # some data handlers need a restart
      if callable(getattr(validds,'start_epoch',None)):
         validds.start_epoch()

      # get data iterator for validation
      if callable(getattr(validds,'batch_gen',None)):
         logger.info('using batch_gen method for valid')
         validds_itr = iter(validds.batch_gen())
      else:
         validds_itr = validds

      for batch_data in validds_itr:
         logger.debug('got validation batch %s',batch_counter)
         
         inputs = batch_data[0].to(device)
         targets = batch_data[1].to(device)

         logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

         start_forward = time.time()

         logger.debug('zeroed opt')
         outputs,endpoints = self(inputs)
         logger.debug('got outputs: %s targets: %s',outputs,targets)

         # logger.info('>> pred = %s targets = %s',pred,targets)
         outputs = torch.softmax(outputs,dim=1)
         # logger.info('gt = %s',pred)
         pred = outputs.argmax(dim=1).float()
         # logger.info('argmax = %s',pred)

         eq = torch.eq(pred,targets.float())
         # logger.info('eq = %s',eq)

         accuracy = torch.sum(eq).float() / float(targets.shape[0])
         try:
            batch_confmat = confusion_matrix(pred,targets,labels=range(len(config['data_handling']['classes'])))
            confmat += batch_confmat
         except:
            logger.exception('error batch_confmat = \n %s confmat = \n %s pred = \n %s targets = \n%s',batch_confmat,confmat,pred,targets)

         end = time.time()

         data_time.add_value(start_forward - start_data)
         forward_time.add_value(end - start_forward)
         batch_time.add_value(end - start_data)

         batch_counter += 1

         # print statistics
         if config['rank'] == 0 and batch_counter % status == 0:
            mean_img_per_second = (forward_time.calc_mean() + backward_time.calc_mean()) / batch_size
            mean_img_per_second = 1. / mean_img_per_second
            
            logger.info('<[%5d of %5d]> valid accuracy: %6.4f images/sec: %6.2f   data time: %6.3f  forward time: %6.3f confmat = %s',batch_counter,len(validds),accuracy,mean_img_per_second,data_time.calc_mean(),forward_time.calc_mean(),confmat)
            logger.info('prediction = %s',pred)

         start_data = time.time()

   @staticmethod
   def get_loss(config):

      if 'loss' not in config:
         raise Exception('must include "loss" section in config file')

      config = config['loss']

      if 'func' not in config:
         raise Exception('must include "func" loss section in config file')

      if 'CrossEntropyLoss' in config['func']:
         weight = None
         if 'weight' in config:
            weight = config['loss_weight']
         size_average = None
         if 'size_average' in config:
            size_average = config['loss_size_average']
         ignore_index = -100
         if 'ignore_index' in config:
            ignore_index = config['loss_ignore_index']
         reduce = None
         if 'reduce' in config:
            reduce = config['loss_reduce']
         reduction = 'mean'
         if 'reduction' in config:
            reduction = config['loss_reduction']

         return torch.nn.CrossEntropyLoss(weight,size_average,ignore_index,reduce,reduction)
      if 'pointnet_class_loss' in config['func']:
         return PointNet1d.pointnet_class_loss
      else:
         raise Exception('%s loss function is not recognized' % config['func'])

   @staticmethod
   def get_accuracy(config):
      if 'CrossEntropyLoss' in config['loss']['func'] or 'pointnet_class_loss' in config['loss']['func']:
         
         return PointNet1d.multiclass_acc
      else:
         if 'func' not in config['model']:
            raise Exception('loss function not defined in config')
         else:
            raise Exception('%s loss function is not recognized' % config['loss']['func'])

   @staticmethod
   def pointnet_class_loss(pred,targets,end_points,reg_weight=0.001,device='cpu'):
      criterion = torch.nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
      classify_loss = criterion(pred, targets)
      # print('prediction = %s' % torch.nn.Softmax()(pred) )
      
      # Enforce the transformation as orthogonal matrix
      mat_loss = 0
      # if 'input_trans' in end_points:
      #    tran = end_points['input_trans']

      #    diff = torch.mean(torch.bmm(tran, tran.permute(0, 2, 1)), 0)
      #    mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1]))

      if 'feature_trans' in end_points:
         tran = end_points['feature_trans']

         diff = torch.mean(torch.bmm(tran, tran.permute(0, 2, 1)), 0)
         mat_loss += torch.nn.MSELoss()(diff, torch.eye(tran.shape[1],device=device))

      # print('criterion = %s mat_loss = %s' % (classify_loss.item(),mat_loss.item()))
      loss = classify_loss + mat_loss * reg_weight

      return loss

   @staticmethod
   def multiclass_acc(pred,targets):

      # logger.info('>> pred = %s targets = %s',pred,targets)
      pred = torch.softmax(pred,dim=1)
      # logger.info('gt = %s',pred)
      pred = pred.argmax(dim=1).float()
      # logger.info('argmax = %s',pred)

      eq = torch.eq(pred,targets.float())
      # logger.info('eq = %s',eq)

      return torch.sum(eq).float() / float(targets.shape[0])


class Transform1d(torch.nn.Module):
   """ Input (XYZ) Transform Net, input is BxNxK gray image
        Return:
            Transformation matrix of size KxK """
   def __init__(self,height,width):
      super(Transform1d,self).__init__()

      self.width = width

      self.conv64 = utils.Conv1d(width,64,pool=False)
      self.conv128 = utils.Conv1d(64,128,pool=False)
      self.conv1024 = utils.Conv1d(128,1024,pool=False)
      
      self.pool = torch.nn.MaxPool1d(height)
      
      self.linear512 = utils.Linear(1024,512)
      self.linear256 = utils.Linear(512,256)

      self.linearK = torch.nn.Linear(256,width * width)
      self.linearK.bias = torch.nn.Parameter(torch.eye(width).view(width * width))

      #self.weights = torch.zeros(256,width * width,requires_grad=True)
      #self.biases  = torch.eye(width,requires_grad=True).flatten()

   def forward(self,x):
      batch_size = x.shape[0]
      
      x = self.conv64(x)
      x = self.conv128(x)
      x = self.conv1024(x)
      
      x = self.pool(x)
      
      x = x.reshape([batch_size,-1])
      x = self.linear512(x)
      x = self.linear256(x)
      
      #x = torch.matmul(x,self.weights) + self.biases
      x = self.linearK(x)
      x = x.reshape([batch_size,self.width,self.width])
      
      return x
