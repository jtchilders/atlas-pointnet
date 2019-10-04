import pytorch.optimizer as opt
import pytorch.loss as losses
import numpy as np
from sklearn.metrics import confusion_matrix
import torch,logging,time
import CalcMean,psutil
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
   elif 'pointnet1d_semseg' in config['model']['model']:
      logger.info('using pointnet model with point-wise labeling')
      
      import pytorch.pointnet as pointnet
      model = pointnet.PointNet1d_SemSeg(config)
      model.float()
      model.to(device)
      return model
   elif 'pointnet1d' in config['model']['model']:
      logger.info('using pointnet model')
      
      import pytorch.pointnet as pointnet
      model = pointnet.PointNet1d(config)
      model.float()
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


def train_model(model,opt,lrsched,trainds,validds,config,writer=None):

   batch_size = config['training']['batch_size']
   status = config['status']
   epochs = config['training']['epochs']
   # nval = config['nval']
   nval_tests = config['nval_tests']
   nsave = config['nsave']
   model_save = config['model_save']
   rank = config['rank']

   if 'mean_class_iou' in config['loss']['acc']:
      classes = config['data_handling']['classes']
      nclasses = len(classes)
      class_accuracy = [CalcMean.CalcMean() for _ in classes]

   # some data handlers need a restart
   if 'csv_pool' == config['data_handling']['input_format']:
      validds.start_epoch()

      # get data iterator for validation
      logger.info('using batch_gen method for valid')
      validds_itr = iter(validds.batch_gen())
   else:
      validds_itr = validds

   loss = losses.get_loss(config)
   acc = losses.get_accuracy(config)

   data_time = CalcMean.CalcMean()
   move_time = CalcMean.CalcMean()
   forward_time = CalcMean.CalcMean()
   backward_time = CalcMean.CalcMean()
   batch_time = CalcMean.CalcMean()
   loss_time = CalcMean.CalcMean()
   acc_time = CalcMean.CalcMean()

   monitor_loss = CalcMean.CalcMean()
   monitor_acc = CalcMean.CalcMean()

   valid_batch_counter = 0
   for epoch in range(epochs):
      logger.info(' epoch %s of %s',epoch,epochs)

      # some data handlers need a restart
      if 'csv_pool' == config['data_handling']['input_format']:
         trainds.start_epoch()

         # get data iterator
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

      model.train()
      start_data = time.time()
      for batch_counter,(inputs,weights,targets) in enumerate(trainds_itr):
         end_data = time.time()

         # logger.info('inputs: %s targets: %s',inputs.shape,targets.shape)
         # logger.info('inputs: %s targets: %s',inputs[0,...,0],targets[0,0])

         if inputs.shape[0] != batch_size:
            logger.warning('input has incorrect batch size: %s',inputs.shape)
            continue

         if targets.shape[0] != batch_size:
            logger.warning('target has incorrect batch size: %s',targets.shape)
            continue
         
         # logger.debug('got training batch %s',batch_counter)
         start_move = end_data
         inputs = inputs.to(device)
         weights = weights.to(device)
         targets = targets.to(device)
         end_move = time.time()

         opt.zero_grad()
         # logger.debug('zeroed opt')
         start_forward = time.time()
         outputs,endpoints = model(inputs)
         # logger.info('outputs = %s targets = %s',torch.nn.Softmax(dim=1)(outputs)[0,...,0],targets[0,0])
         end_forward = time.time()
         # logger.debug('got outputs: %s targets: %s',outputs,targets)

         start_loss = end_forward
         loss_value = loss(outputs,targets,endpoints,weights,device=device)
         end_loss = time.time()
         monitor_loss.add_value(loss_value)
         # logger.debug('got loss')

         start_acc = time.time()
         acc_value = acc(outputs,targets,device)
         if 'mean_class_iou' in config['loss']['acc']:
            for i in range(nclasses):
               class_accuracy[i].add_value(acc_value[i])
            monitor_acc.add_value(acc_value.mean())
         else:
            monitor_acc.add_value(acc_value)
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
         if batch_counter % status == 0:
            mean_img_per_second = (forward_time.calc_mean() + backward_time.calc_mean()) / batch_size
            mean_img_per_second = 1. / mean_img_per_second
            
            logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f train acc: %6.4f  images/sec: %6.2f   data time: %6.3f move time: %6.3f forward time: %6.3f loss time: %6.3f  backward time: %6.3f acc time: %6.3f inclusive time: %6.3f',epoch + 1,epochs,batch_counter,len(trainds),monitor_loss.calc_mean(),monitor_acc.calc_mean(),mean_img_per_second,data_time.calc_mean(),move_time.calc_mean(),forward_time.calc_mean(),acc_time.calc_mean(),backward_time.calc_mean(),acc_time.calc_mean(),batch_time.calc_mean())
            if 'mean_class_iou' in config['loss']['acc']:
               logger.info('<[%3d of %3d, %5d of %5d]> class accuracy: %s',epoch + 1,epochs,batch_counter,len(trainds),['%6.4f' % x.calc_mean() for x in class_accuracy])
            # mem = psutil.virtual_memory()
            # logger.info('<[%3d of %3d, %5d of %5d]> cpu usage: %s mem total: %s mem free: %s (%4.1f%%)',
            #    epoch + 1,epochs,batch_counter,len(trainds),psutil.cpu_percent(),mem.total,
            #    mem.free,mem.free / mem.total * 100.)
            # logger.info('running count = %s',trainds.running_class_count)
            # logger.info('prediction = %s',torch.nn.Softmax(dim=1)(outputs))

            if writer and rank == 0:
               global_batch = epoch * len(trainds) + batch_counter
               writer.add_scalars('loss',{'train':monitor_loss.calc_mean()},global_batch)
               writer.add_scalars('accuracy',{'train':monitor_acc.calc_mean()},global_batch)
               if 'mean_class_iou' in config['loss']['acc']:
                  for i in range(nclasses):
                     writer.add_scalars('accuracy_%s' % i,{'train':class_accuracy[i].calc_mean()},global_batch)
               writer.add_scalar('image_per_second',mean_img_per_second,global_batch)

            monitor_loss = CalcMean.CalcMean()
            monitor_acc = CalcMean.CalcMean()

         # periodically save the model
         if batch_counter % nsave == 0 and rank == 0:
            torch.save(model.state_dict(),model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))
         
         # restart data timer
         start_data = time.time()

         # end training loop, check if should exit
         # if batch_limiter is not None and batch_counter > batch_limiter:
         #    break
      
      logger.info('epoch %s complete, running validation on %s batches',epoch,nval_tests)
      # if this is set, skip validation
      # if batch_limiter is not None:
      #    break

      # every epoch, evaluate validation data set
      model.eval()

      # if validds.reached_end:
      #    logger.warning('restarting validation file pool.')
      #    validds.start_file_pool(1)

      vloss = CalcMean.CalcMean()
      vacc = CalcMean.CalcMean()
      if 'mean_class_iou' in config['loss']['acc']:
         vclass_acc = [CalcMean.CalcMean() for _ in range(nclasses)]

      for valid_batch_counter,(inputs,weights,targets) in enumerate(validds_itr):
         logger.info('validation batch %s of %s',valid_batch_counter,len(validds))

         inputs = inputs.to(device)
         targets = targets.to(device)
         weights = weights.to(device)

         outputs,endpoints = model(inputs)

         loss_value = loss(outputs,targets,endpoints,weights,device)
         vloss.add_value(loss_value.item())
         acc_value = acc(outputs,targets,device)
         if 'mean_class_iou' in config['loss']['acc']:
            for i in range(nclasses):
               vclass_acc[i].add_value(acc_value[i])
         else:
            vacc.add_value(acc_value.item())
         
         if valid_batch_counter > nval_tests:
            break


      mean_acc = vacc.calc_mean()
      mean_loss = vloss.calc_mean()
      if config['hvd'] is not None:
         mean_acc  = config['hvd'].allreduce(torch.tensor([mean_acc]))
         mean_loss = config['hvd'].allreduce(torch.tensor([mean_loss]))
      

      # add validation to tensorboard
      if writer and rank == 0:
         global_batch = epoch * len(trainds) + batch_counter
         writer.add_scalars('loss',{'valid':mean_loss},global_batch)
         writer.add_scalars('accuracy',{'valid':mean_acc},global_batch)
         if 'mean_class_iou' in config['loss']['acc']:
            for i in range(nclasses):
               writer.add_scalars('accuracy_%s' % i,{'valid':vclass_acc[i].calc_mean()},global_batch)
         
      logger.info('>[%3d of %3d, %5d of %5d]<<< ave valid loss: %6.4f ave valid acc: %6.4f on %s batches >>>',epoch + 1,epochs,batch_counter,len(trainds),mean_loss,mean_acc,valid_batch_counter+1)
      if 'mean_class_iou' in config['loss']['acc']:
         logger.info('>[%3d of %3d, %5d of %5d]<<< valid class acc: %s',epoch + 1,epochs,batch_counter,len(trainds),['%6.4f' % x.calc_mean() for x in vclass_acc])

      model.train()


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
   start_data = time.time()

   confmat = np.zeros((nClasses,nClasses))

   loss = losses.get_loss(config)
   acc = losses.get_accuracy(config)
   
   # some data handlers need a restart
   if 'csv_pool' == config['data_handling']['input_format']:
      validds.start_epoch()

      # get data iterator for validation
      logger.info('using batch_gen method for valid')
      validds_itr = iter(validds.batch_gen())
   else:
      validds_itr = validds

   for batch_counter,(inputs,targets) in enumerate(validds_itr):
      logger.debug('got validation batch %s',batch_counter)
      
      inputs = inputs.to(device)
      targets = targets.to(device)

      logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

      start_forward = time.time()

      logger.debug('zeroed opt')
      pred,_ = net(inputs)
      logger.debug('got pred: %s targets: %s',pred.shape,targets.shape)

      logger.debug(' pred = %s targets = %s',pred[0,:,0],targets[0,0])
      pred = torch.softmax(pred,dim=1)
      logger.debug('softmax = %s',pred[0,:,0])
      pred = pred.argmax(dim=1).float()
      logger.debug('argmax = %s',pred[0,0])

      eq = torch.eq(pred,targets.float())
      logger.debug('eq = %s',eq[0,0])

      accuracy = torch.sum(eq).float() / float(targets.shape.numel())
      logger.debug('acc = %s',accuracy)
      try:
         batch_confmat = confusion_matrix(pred.reshape(-1),targets.reshape(-1),labels=range(len(config['data_handling']['classes'])))
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
