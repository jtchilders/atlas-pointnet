import pytorch.optimizer as opt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch,logging,time
import CalcMean
logger = logging.getLogger(__name__)
#torch.set_printoptions(sci_mode=False,precision=3)

# detect device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model(config):

   config['device'] = device
   logger.info('device:             %s',device)

   if 'pointnet2d' in config['model']['model']:
      logger.info('using pointnet model')
      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 2)

      nChannels = 1  # input_shape[0]
      nPoints = input_shape[0]
      nCoords = input_shape[1]
      nClasses = len(config['data_handling']['classes'])
      logger.debug('nChannels = %s, nPoints = %s, nCoords = %s, nClasses = %s',nChannels,nPoints,nCoords,nClasses)
      import pytorch.pointnet as pointnet
      model = pointnet.PointNet2d(nChannels,nPoints,nCoords,nClasses)
      model.to(device)
      return model
   elif 'pointnet1d' in config['model']['model']:
      logger.info('using pointnet model')
      
      import pytorch.pointnet as pointnet
      model = pointnet.PointNet1d(config)
      model.to(device)
      return model
   elif 'yolo_classonly' in config['model']['model']:
      logger.info('using yolo_classonly model')
      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 3)

      import pytorch.yolo_classonly as yolo
      model = yolo.YOLOClassOnly(config)
      model.to(device)
      return model
   else:
      raise Exception('no model specified')


def setup(net,hvd,config):

   # get optimizer
   optimizer = opt.get_optimizer(net,config)

   if 'lrsched' in config['optimizer']:
      lrsched = opt.get_scheduler(optimizer,config)

   if config['rank'] == 0 and config['input_model_pars']:
      logger.info('loading model pars from file %s',config['input_model_pars'])
      net.load_state_dict(torch.load(config['input_model_pars'],map_location=lambda storage, loc: storage))

   if config['horovod']:
      logger.info('hvd broadcast')
      hvd.broadcast_parameters(net.state_dict(),root_rank=0)

      optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=net.named_parameters())

   return optimizer,lrsched


def train_model(net,opt,loss,acc,lrsched,trainds,validds,config,writer=None):

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
      validds_itr = validds.batch_gen
   else:
      validds_itr = validds

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
         train_itr = trainds.batch_gen
      else:
         train_itr = trainds

      if lrsched:
         lrsched.step()

      for param_group in opt.param_groups:
         logging.info('learning rate: %s',param_group['lr'])
         if writer:
            writer.add_scalar('learning_rate',param_group['lr'],epoch)

      net.train()
      net.to(device)
      batch_counter = 0
      start_data = time.time()
      for batch_data in train_itr:
         end_data = time.time()
         
         # logger.debug('got training batch %s',batch_counter)
         start_move = end_data
         inputs = batch_data[0]
         #inputs = inputs.to(device)
         targets = batch_data[1]
         #targets = targets.to(device)
         end_move = time.time()

         # logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

         opt.zero_grad()
         # logger.debug('zeroed opt')
         start_forward = time.time()
         outputs,endpoints = net(inputs)
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
               net.eval()

               # if validds.reached_end:
               #    logger.warning('restarting validation file pool.')
               #    validds.start_file_pool(1)

               for _ in range(nval_tests):

                  try:
                     batch_data = next(validds_itr)
                     valid_batch_counter += 1
                  except StopIteration:
                     validds.start_epoch()
                     validds_itr = validds.batch_gen()
                     batch_data = next(validds_itr)
                     valid_batch_counter = 0
                  
                  inputs = batch_data[0]  # .to(device)
                  targets = batch_data[1]  # .to(device)

                  # logger.debug('valid inputs: %s targets: %s',inputs.shape,targets.shape)

                  outputs,endpoints = net(inputs)
                  #true_positive_accuracy,true_or_false_positive_accuracy,filled_grids_accuracy = accuracyCalc.eval_acc(outputs,targets,inputs)

                  loss_value = loss(outputs,targets,endpoints,device=device)
                  acc_value = acc(outputs,targets)

                  if writer:
                     global_batch = epoch * len(trainds) + batch_counter
                     writer.add_scalars('loss',{'valid':loss_value.item()},global_batch)
                     writer.add_scalars('accuracy',{'valid':acc_value.item()},global_batch)

                  logger.info('>[%3d of %3d, %5d of %5d]< valid loss: %6.4f valid acc: %6.4f',epoch + 1,epochs,batch_counter,len(trainds),loss_value.item(),acc_value.item())
                  
               net.train()

            if batch_counter % nsave == 0:
               torch.save(net.state_dict(),model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

         start_data = time.time()


def valid_model(net,validds,config):

   batch_size = config['training']['batch_size']
   status = config['status']
   nClasses = len(config['data_handling']['classes'])

   data_time = CalcMean.CalcMean()
   forward_time = CalcMean.CalcMean()
   backward_time = CalcMean.CalcMean()
   batch_time = CalcMean.CalcMean()

   net.eval()
   net.to(device)
   batch_counter = 0
   start_data = time.time()

   confmat = np.zeros((nClasses,nClasses))

   for batch_data in validds.batch_gen():
      logger.debug('got validation batch %s',batch_counter)
      
      inputs = batch_data[0].to(device)
      targets = batch_data[1].to(device)

      logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

      start_forward = time.time()

      logger.debug('zeroed opt')
      outputs,endpoints = net(inputs)
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
