import torch,time
import pytorch.utils as utils
import logging
import CalcMean
from pytorch.model import device
logger = logging.getLogger(__name__)


class YOLO_PRECONNECT(torch.nn.Module):
   def __init__(self,channels):
      super(YOLO_PRECONNECT,self).__init__()

      self.conv1 = utils.Conv2d(channels,64,(3,3),padding=1,pool=True,pool_size=(2,4))
      self.conv2 = utils.Conv2d(64,128,(3,3),padding=1,pool=False)
      self.conv3 = utils.Conv2d(128,256,(3,3),padding=1,pool=False)
      self.conv4 = utils.Conv2d(256,512,(3,3),padding=1,pool=True,pool_size=(2,4))
      self.conv5 = utils.Conv2d(512,256,(1,1),pool=False)
      self.conv6 = utils.Conv2d(256,512,(3,3),padding=1,pool=False)
      self.conv7 = utils.Conv2d(512,256,(1,1),pool=False)
      self.conv8 = utils.Conv2d(256,512,(3,3),padding=1,pool=True,pool_size=(2,4))

      self.output_size = 512
      self.reduction_factor = (2 * 2 * 2,4 * 4 * 4)

   def forward(self,x):

      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)
      x = self.conv6(x)
      x = self.conv7(x)
      x = self.conv8(x)

      return x


class YOLO_POSTCONNECT(torch.nn.Module):
   def __init__(self,channels):
      super(YOLO_POSTCONNECT,self).__init__()

      self.conv1 = utils.Conv2d(channels,1024,(3,3),padding=1,pool=False)
      self.conv2 = utils.Conv2d(1024,512,(1,1),pool=False)
      self.conv3 = utils.Conv2d(512,1024,(3,3),padding=1,pool=False)
      self.conv4 = utils.Conv2d(1024,512,(1,1),pool=False)
      self.conv5 = utils.Conv2d(512,1024,(3,3),padding=1,pool=False)

      self.output_size = 1024
      self.reduction_factor = (1,1)

   def forward(self,x):

      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)
      
      return x


class YOLOClassOnly(torch.nn.Module):
   def __init__(self,config):
      super(YOLOClassOnly,self).__init__()

      img_shape = config['data_handling']['image_shape']
      num_classes = len(config['data_handling']['classes'])

      self.pre_connect = YOLO_PRECONNECT(img_shape[0])

      self.post_connect = YOLO_POSTCONNECT(self.pre_connect.output_size)

      self.conv_connect = utils.Conv2d(self.pre_connect.output_size + self.post_connect.output_size,1024,(3,3),padding=1,pool=False)

      self.feature_layer = utils.Conv2d(1024,1 + num_classes,(3,3),padding=1,pool=False)

      reduction_factor = (self.pre_connect.reduction_factor[0] * self.post_connect.reduction_factor[0],
                          self.pre_connect.reduction_factor[1] * self.post_connect.reduction_factor[1])

      gridh = img_shape[1] // reduction_factor[0]
      gridw = img_shape[2] // reduction_factor[1]

      self.output_grid = (gridh,gridw)
      logger.info('grid size = %s x %s',self.output_grid[0],self.output_grid[1])

   def forward(self,x):

      connection = self.pre_connect(x)

      x = self.post_connect(connection)

      x = torch.cat((x,connection),dim=1)

      x = self.conv_connect(x)

      x = self.feature_layer(x)

      return x

   def train_model(self,opt,lrsched,trainds,validds,config,writer=None):
      self.float()
      batch_size = config['training']['batch_size']
      status = config['status']
      epochs = config['training']['epochs']
      nval = config['nval']
      nval_tests = config['nval_tests']
      nsave = config['nsave']
      model_save = config['model_save']

      validds_itr = iter(validds)

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

         if lrsched:
            lrsched.step()

         for param_group in opt.param_groups:
            logging.info('learning rate: %s',param_group['lr'])
            if writer:
               writer.add_scalar('learning_rate',param_group['lr'],epoch)

         self.train()
         self.to(device)
         #batch_counter = 0
         start_data = time.time()
         for batch_counter,batch_data in enumerate(trainds):
            end_data = time.time()
            
            # logger.debug('got training batch %s',batch_counter)
            start_move = end_data
            inputs = batch_data[0].to(device)
            #inputs = inputs.to(device)
            targets = batch_data[1].to(device)
            #targets = targets.to(device)
            end_move = time.time()

            #print('inputs: %s targets: %s' % (inputs.dtype,targets.dtype))

            opt.zero_grad()
            # logger.debug('zeroed opt')
            start_forward = time.time()
            outputs = self(inputs)
            end_forward = time.time()
            # logger.debug('got outputs: %s targets: %s',outputs,targets)

            start_loss = end_forward
            loss_value = self.grid_id_loss(outputs,targets)
            end_loss = time.time()
            monitor_loss.add_value(loss_value)
            # logger.debug('got loss')

            start_acc = time.time()
            acc_value = self.grid_id_accuracy(outputs,targets)
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

            #batch_counter += 1

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

                  for _ in range(nval_tests):

                     try:
                        batch_data = next(validds_itr)
                        valid_batch_counter += 1
                     except StopIteration:
                        validds_itr = iter(validds)
                        batch_data = next(validds_itr)
                        valid_batch_counter = 0
                     
                     inputs = batch_data[0]  # .to(device)
                     targets = batch_data[1]  # .to(device)

                     outputs = self(inputs)

                     loss_value = self.grid_id_loss(outputs,targets)
                     acc_value = self.grid_id_accuracy(outputs,targets)

                     if writer:
                        global_batch = epoch * len(trainds) + batch_counter
                        writer.add_scalars('loss',{'valid':loss_value.item()},global_batch)
                        writer.add_scalars('accuracy',{'valid':acc_value.item()},global_batch)

                     logger.info('>[%3d of %3d, %5d of %5d]< valid loss: %6.4f valid acc: %6.4f',epoch + 1,epochs,batch_counter,len(trainds),loss_value.item(),acc_value.item())
                     
                  self.train()

               if batch_counter % nsave == 0:
                  torch.save(self.state_dict(),model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

            start_data = time.time()

   @staticmethod
   def grid_id_loss(outputs,targets):
      pred_grid_conf = outputs[:,0,...].float()
      true_grid_conf = targets[:,0,...].float()

      # print('pred_grid_conf = ',pred_grid_conf.shape,'true_grid_conf=',true_grid_conf.shape)
      # agreement of grid level object exits label
      # grid_id_loss = torch.sum( (true_grid_conf - pred_grid_conf) ** 2 )
      pos_weight = torch.Tensor(pred_grid_conf.size(1),pred_grid_conf.size(2)).fill_(pred_grid_conf.size(1) * pred_grid_conf.size(2))
      grid_id_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred_grid_conf,true_grid_conf)

      return grid_id_loss

   @staticmethod
   def grid_id_accuracy(outputs,targets):
      grid_pred = outputs[:,0,...]  # [B,gy,gx]
      grid_true = targets[:,0,...]
      # print('grid_pred = ',grid_pred.shape,'grid_true=',grid_true.shape)

      grid_pred = torch.ge(grid_pred,0.5).int()

      correct_pred = torch.eq(grid_pred,grid_true).int() * grid_true  # [B,gy,gx]

      total_correct_pred = correct_pred.sum(dim=(1,2),dtype=torch.float)
      total_true = grid_true.sum(dim=(1,2),dtype=torch.float)  # [B]
      
      # print('correct_pred = ',total_correct_pred,'total_true=',total_true)
      fraction_pred = total_correct_pred / total_true
      # print('fp = ',fraction_pred)
      return fraction_pred.mean()
