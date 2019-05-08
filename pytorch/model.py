import pytorch.pointnet as pointnet
import pytorch.optimizer as opt
import torch,logging,time
import CalcMean
logger = logging.getLogger(__name__)


def get_model(config):

   if 'pointnet' in config['model']['model']:
      logger.info('using pointnet model')
      input_shape = config['data_handling']['image_shape']

      assert(len(input_shape) == 2)

      nChannels = 1 #input_shape[0]
      nPoints = input_shape[0]
      nCoords = input_shape[1]
      nClasses = len(config['data_handling']['classes'])
      logger.debug('nChannels = %s, nPoints = %s, nCoords = %s, nClasses = %s',nChannels,nPoints,nCoords,nClasses)
      model = pointnet.PointNet(nChannels,nPoints,nCoords,nClasses)

      return model.float()
   else:
      raise Exception('no model specified')


def setup(net,hvd,config):

   # get optimizer
   optimizer = opt.get_optimizer(net,config)

   if config['rank'] == 0 and config['input_model_pars']:
      logger.info('loading model pars from file %s',config['input_model_pars'])
      net.load_state_dict(torch.load(config['input_model_pars']))

   if config['horovod']:
      logger.info('hvd broadcast')
      hvd.broadcast_parameters(net.state_dict(),root_rank=0)

      optimizer = hvd.DistributedOptimizer(optimizer,named_parameters=net.named_parameters())

   return optimizer


def train_model(net,opt,loss,acc,lrsched,trainds,validds,config):

   batch_size = config['training']['batch_size']
   status = config['status']
   epochs = config['training']['epochs']
   nval = config['nval']
   nval_tests = config['nval_tests']
   nsave = config['nsave']
   model_save = config['model_save']

   validds_itr = validds.batch_gen()

   data_time = CalcMean.CalcMean()
   forward_time = CalcMean.CalcMean()
   backward_time = CalcMean.CalcMean()
   batch_time = CalcMean.CalcMean()

   valid_batch_counter = 0
   for epoch in range(epochs):
      logger.info(' epoch %s of %s',epoch,epochs)
      
      if lrsched:
         lrsched.step()

      for param_group in opt.param_groups:
         logging.info('learning rate: %s',param_group['lr'])

      net.train()
      batch_counter = 0
      start_data = time.time()
      for batch_data in trainds.batch_gen():
         logger.debug('got training batch %s',batch_counter)
         
         inputs = batch_data[0]
         targets = batch_data[1]

         logger.debug('inputs: %s targets: %s',inputs.shape,targets.shape)

         start_forward = time.time()

         opt.zero_grad()
         logger.debug('zeroed opt')
         outputs,endpoints = net(inputs)
         logger.debug('got outputs: %s targets: %s',outputs,targets)

         loss_value = loss(outputs,targets,endpoints)
         logger.debug('got loss')

         acc_value = acc(outputs,targets)

         start_backward = time.time()
         loss_value.backward()
         opt.step()

         end = time.time()

         data_time.add_value(start_forward - start_data)
         forward_time.add_value(start_backward - start_forward)
         backward_time.add_value(end - start_backward)
         batch_time.add_value(end - start_data)

         batch_counter += 1

         # print statistics
         if config['rank'] == 0:
            if batch_counter % status == 0:
               mean_img_per_second = (forward_time.calc_mean() + backward_time.calc_mean()) / batch_size
               mean_img_per_second = 1. / mean_img_per_second
               
               logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f train acc: %6.4f  images/sec: %6.2f   data time: %6.3f  forward time: %6.3f  backward time: %6.3f',epoch + 1,epochs,batch_counter,len(trainds),loss_value.item(),acc_value.item(),mean_img_per_second,data_time.calc_mean(),forward_time.calc_mean(),backward_time.calc_mean())

            if batch_counter % nval == 0:
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
                     validds_itr = validds.batch_gen()
                     batch_data = next(validds_itr)
                     valid_batch_counter = 0
                  
                  inputs = batch_data[0]
                  targets = batch_data[1]

                  logger.debug('valid inputs: %s targets: %s',inputs.shape,targets.shape)

                  outputs,endpoints = net(inputs)
                  #true_positive_accuracy,true_or_false_positive_accuracy,filled_grids_accuracy = accuracyCalc.eval_acc(outputs,targets,inputs)

                  loss_value = loss(outputs,targets,endpoints)
                  acc_value = acc(outputs,targets)

                  logger.info('>[%3d of %3d, %5d of %5d]< valid loss: %6.4f valid acc: %6.4f',epoch + 1,epochs,batch_counter,len(trainds),loss_value.item(),acc_value.item())
                  
               net.train()

            if batch_counter % nsave == 0:
               torch.save(net.state_dict(),model_save + '_%05d_%05d.torch_model_state_dict' % (epoch,batch_counter))

         start_data = time.time()


