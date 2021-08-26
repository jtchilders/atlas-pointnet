#!/usr/bin/env python3
import argparse,logging,socket,json,sys,psutil,time,os
import numpy as np
import data_handler
import model
import torch
import loss
import accuracy
import optimizer
import tensorboardX
import CalcMean
import multiprocessing as mp

logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config_file',help='configuration file in json format',required=True)
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--model_save',help='base name of saved model parameters for later loading')
   parser.add_argument('--nsave',default=100,type=int,help='frequency in batch number to save model')

   parser.add_argument('--nval',default=100,type=int,help='frequency to evaluate validation sample in batch numbers')
   parser.add_argument('--nval_tests',default=-1,type=int,help='number batches to test per validation run')

   parser.add_argument('--status',default=20,type=int,help='frequency to print loss status in batch numbers')

   parser.add_argument('--batch',default=-1,type=int,help='set batch size, overrides file config')

   parser.add_argument('--random_seed',default=0,type=int,help='numpy random seed')

   parser.add_argument('--valid_only',default=False,action='store_true',help='flag that triggers validation run. prints confusion matrix.')

   parser.add_argument('--batch_limiter',help='if set to an integer, will limit the number of batches during training. Use this to create short training runs for profiling.',type=int)

   parser.add_argument('-i','--input_model_pars',help='if provided, the file will be used to fill the models state dict from a previous run.')
   parser.add_argument('-e','--epochs',type=int,default=-1,help='number of epochs')
   parser.add_argument('-l','--logdir',help='log directory for tensorboardx')

   parser.add_argument('--horovod',default=False, action='store_true', help="Setup for distributed training")

   parser.add_argument('--cpu-only',default=False, action='store_true', help='set to force CPU only running')


   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   log_level = logging.INFO

   if args.debug and not args.error and not args.warning:
      log_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      log_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      log_level = logging.WARNING


   rank = 0
   nranks = 1
   local_rank = 0
   local_size = 1
   hvd = None
   if args.horovod:
      print('importing horovod')
      import horovod.torch as hvd
      print('imported horovod')
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      local_rank = hvd.local_rank()
      local_size = hvd.local_size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if rank > 0 and log_level == logging.INFO:
      log_level = logging.WARNING

   logging.basicConfig(level=log_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   device = torch.device('cpu')
   if torch.cuda.is_available() and not args.cpu_only:
      device = torch.device('cuda:%d' % local_rank)
      torch.cuda.set_device(device)

   model_save = args.model_save
   if model_save is None:
      model_save = os.path.join(args.logdir,'model')

   logger.warning('rank %6s of %6s    local rank %6s of %6s',rank,nranks,local_rank,local_size)
   logger.info('hostname:           %s',socket.gethostname())
   logger.info('python version:     %s',sys.version)
   logger.info('num_threads:        %s',torch.get_num_threads())
   logger.info('torch version:      %s',torch.__version__)
   logger.info('torch file:         %s',torch.__file__)
   

   logger.info('config file:        %s',args.config_file)
   logger.info('num files:          %s',args.num_files)
   logger.info('model_save:         %s',model_save)
   logger.info('random_seed:        %s',args.random_seed)
   logger.info('valid_only:         %s',args.valid_only)
   logger.info('nsave:              %s',args.nsave)
   logger.info('nval:               %s',args.nval)
   logger.info('nval_tests:         %s',args.nval_tests)
   logger.info('status:             %s',args.status)
   logger.info('input_model_pars:   %s',args.input_model_pars)
   logger.info('epochs:             %s',args.epochs)
   logger.info('horovod:            %s',args.horovod)
   logger.info('cpu_only:           %s',args.cpu_only)
   logger.info('logdir:             %s',args.logdir)

   np.random.seed(args.random_seed)

   config_file = json.load(open(args.config_file))
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   config_file['input_model_pars'] = args.input_model_pars
   config_file['horovod'] = args.horovod
   config_file['status'] = args.status
   config_file['nval'] = args.nval
   config_file['nval_tests'] = args.nval_tests
   config_file['nsave'] = args.nsave
   config_file['model_save'] = model_save
   config_file['valid_only'] = args.valid_only
   config_file['batch_limiter'] = args.batch_limiter
   config_file['cpu_only'] = args.cpu_only

   if args.valid_only and not args.input_model_pars:
      logger.error('if valid_only set, must provide input model')
      return

   if args.batch > 0:
      logger.info('setting batch size from command line: %s', args.batch)
      config_file['training']['batch_size'] = args.batch
   if args.epochs > 0:
      logger.info('setting epochs from command line: %s', args.epochs)
      config_file['training']['epochs'] = args.epochs

   logger.info('configuration = \n%s',json.dumps(config_file, indent=4, sort_keys=True))
   config_file['hvd'] = hvd

   # get datasets for training and validation
   trainds,testds = data_handler.get_datasets(config_file)

   # setup tensorboard
   writer = None
   if args.logdir and rank == 0:
      if not os.path.exists(args.logdir):
         os.makedirs(args.logdir)
      writer = tensorboardX.SummaryWriter(args.logdir)
   
   logger.info('building model')
   torch.manual_seed(args.random_seed)

   net = model.get_model(config_file)

   logger.info('model = \n %s',net)

   total_params = sum(p.numel() for p in net.parameters())
   logger.info('trainable parameters: %s',total_params)

   if args.valid_only:
      valid_model(net,validds,config_file)
   else:
      train_model(net,trainds,testds,config_file,device,writer)


def train_model(model,trainds,testds,config,device,writer=None):
   batch_size = config['data']['batch_size']
   status = config['training']['status']
   epochs = config['training']['epochs']
   balanced_loss = config['loss']['balanced']
   # nval = config['nval']
   nval_tests = config['nval_tests']
   nsave = config['nsave']
   model_save = config['model_save']
   rank = config['rank']
   nranks = config['nranks']
   hvd = config['hvd']

   ## create samplers for these datasets
   train_sampler = torch.utils.data.distributed.DistributedSampler(trainds,nranks,rank,shuffle=True,drop_last=True)
   test_sampler = torch.utils.data.distributed.DistributedSampler(testds,nranks,rank,shuffle=True,drop_last=True)

   ## create data loaders
   train_loader = torch.utils.data.DataLoader(trainds,shuffle=False,
                                            sampler=train_sampler,num_workers=config['data']['num_parallel_readers'],
                                            batch_size=batch_size,persistent_workers=True)
   test_loader = torch.utils.data.DataLoader(testds,shuffle=False,
                                            sampler=test_sampler,num_workers=config['data']['num_parallel_readers'],
                                            batch_size=batch_size,persistent_workers=True)

   loss_func = loss.get_loss(config)
   ave_loss = CalcMean.CalcMean()
   acc_func = accuracy.get_accuracy(config)
   ave_acc = CalcMean.CalcMean()

   opt_func = optimizer.get_optimizer(config)
   opt = opt_func(model.parameters(),**config['optimizer']['args'])

   lrsched_func = optimizer.get_learning_rate_scheduler(config)
   lrsched = lrsched_func(opt,**config['lr_schedule']['args'])

   # Add Horovod Distributed Optimizer
   if hvd:
      opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters())
      
      # Broadcast parameters from rank 0 to all other processes.
      hvd.broadcast_parameters(model.state_dict(), root_rank=0)

   model.to(device)

   for epoch in range(epochs):
      logger.info(' epoch %s of %s',epoch,epochs)

      train_sampler.set_epoch(epoch)
      test_sampler.set_epoch(epoch)
      model.to(device)
      for batch_counter,(inputs,targets,class_weights,nonzero_mask) in enumerate(train_loader):
         
         # move data to device
         inputs = inputs.to(device)
         targets = targets.to(device)
         class_weights = class_weights.to(device)
         nonzero_mask = nonzero_mask.to(device)
         
         # zero grads
         opt.zero_grad()
         outputs,endpoints = model(inputs)
         
         # set the weights
         if balanced_loss:
            weights = class_weights
            nonzero_to_class_scaler = torch.sum(nonzero_mask.type(torch.float32)) / torch.sum(class_weights.type(torch.float32))
         else:
            weights = nonzero_mask
            nonzero_to_class_scaler = torch.ones(1,device=device)

         loss_value = loss_func(outputs,targets.long())
         loss_value = torch.mean(loss_value * weights) * nonzero_to_class_scaler
         
         # backward calc grads
         loss_value.backward()

         # apply grads
         opt.step()

         ave_loss.add_value(float(loss_value.to('cpu')))

         # calc acc
         ave_acc.add_value(float(acc_func(outputs,targets,weights).to('cpu')))
         

         # print statistics
         if batch_counter % status == 0:
            
            logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f acc: %6.4f',epoch + 1,epochs,batch_counter,len(trainds)/nranks/batch_size,ave_loss.mean(),ave_acc.mean())

            if writer and rank == 0:
               global_batch = epoch * len(trainds)/nranks/batch_size + batch_counter
               writer.add_scalars('loss',{'train':ave_loss.mean()},global_batch)
               writer.add_scalars('accuracy',{'train':ave_acc.mean()},global_batch)
               #writer.add_histogram('input_trans',endpoints['input_trans'].view(-1),global_batch)

            ave_loss = CalcMean.CalcMean()
            ave_acc = CalcMean.CalcMean()

         # release tensors for memory
         del inputs,targets,weights,endpoints,loss_value

         if config['batch_limiter'] and batch_counter > config['batch_limiter']:
            logger.info('batch limiter enabled, stop training early')
            break

      # save at end of epoch
      torch.save(model.state_dict(),model_save + '_%05d.torch_model_state_dict' % epoch)
      
      if nval_tests == -1:
         nval_tests = len(testds)/nranks/batch_size
      logger.info('epoch %s complete, running validation on %s batches',epoch,nval_tests)

      model.to(device)
      # every epoch, evaluate validation data set
      with torch.no_grad():

         vloss = CalcMean.CalcMean()
         vacc = CalcMean.CalcMean()

         for valid_batch_counter,(inputs,targets,class_weights,nonzero_mask) in enumerate(test_loader):
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            class_weights = class_weights.to(device)
            nonzero_mask = nonzero_mask.to(device)

            # set the weights
            if balanced_loss:
               weights = class_weights
               nonzero_to_class_scaler = torch.sum(nonzero_mask.type(torch.float32)) / torch.sum(class_weights.type(torch.float32))
            else:
               weights = nonzero_mask
               nonzero_to_class_scaler = torch.ones(1,device=device)

            outputs,endpoints = model(inputs)
            
            loss_value = loss_func(outputs,targets.long())
            loss_value = torch.mean(loss_value * weights) * nonzero_to_class_scaler
            vloss.add_value(float(loss_value.to('cpu')))
            
            # calc acc
            vacc.add_value(float(acc_func(outputs,targets,weights).to('cpu')))

            if valid_batch_counter > nval_tests:
               break

         mean_acc = vacc.mean()
         mean_loss = vloss.mean()
         # if config['hvd'] is not None:
         #    mean_acc  = config['hvd'].allreduce(torch.tensor([mean_acc]))
         #    mean_loss = config['hvd'].allreduce(torch.tensor([mean_loss]))
      
         # add validation to tensorboard
         if writer and rank == 0:
            global_batch = epoch * len(trainds)/nranks/batch_size + batch_counter
            writer.add_scalars('loss',{'valid':mean_loss},global_batch)
            writer.add_scalars('accuracy',{'valid':mean_acc},global_batch)
         
         logger.warning('>[%3d of %3d, %5d of %5d]<<< ave valid loss: %6.4f ave valid acc: %6.4f on %s batches >>>',epoch + 1,epochs,batch_counter,len(trainds)/nranks/batch_size,mean_loss,mean_acc,valid_batch_counter + 1)
         

      # update learning rate
      lrsched.step()        



if __name__ == "__main__":
   main()
