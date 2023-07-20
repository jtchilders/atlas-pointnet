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

# for autograd profile
import contextlib
@contextlib.contextmanager
def dummycontext():
    yield None

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
   parser.add_argument('--horovod-groups',type=str,default=None,help="Optional for horovod distributed mode. Comma separated percentages horovod groups for fusion. Sum of percentage must be 100.")
   parser.add_argument('--horovod-data-barrier',default=False, action='store_true', help="Perform barrier after data loading. Throughput will not include time for dataloading")

   parser.add_argument('--filebase',type=str,default=None,help="Optional filebase directory to be prefixed to the filelist")
   parser.add_argument('--cpu-only',default=False, action='store_true', help='set to force CPU only running')

   parser.add_argument('--device', dest='device', default='cpu', help='If set, use the selected device.')
   parser.add_argument('--bf16', action='store_true', help='Datatype used: bf16')
   parser.add_argument('--mixed-precision', action='store_true', help='NVIDIA auto-mixed-precision. bf16 argument will be ignored when this option is on')
   parser.add_argument('--channels_last', action='store_true', help='Enable channels last format')
   parser.add_argument('--profile', dest='profile', default=False, action='store_true', help="Enable Autograd profiling. Generate timeline_X.json files for each epoch")
   parser.add_argument('--num-parallel-readers',type=int,default=-1,help='number of workers for data loader. value>=0 overrides the value from config file')
   parser.add_argument('--train-filelist',type=str,default=None,help="path to the train file list. This will override the one defined in config file")
   parser.add_argument('--test-filelist',type=str,default=None,help="path to the test file list. This will override the one defined in config file")

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

   device = torch.device('cpu')
   if args.device == "xpu" and not args.cpu_only:
      import intel_extension_for_pytorch as ipex

   rank = 0
   nranks = 1
   local_rank = 0
   local_size = 1
   hvd = None
   if args.horovod:
      import horovod.torch as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      local_rank = hvd.local_rank()
      local_size = hvd.local_size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if args.device == "xpu" and not args.cpu_only:
      device = torch.device('xpu:%d' % local_rank)
   elif torch.cuda.is_available() and not args.cpu_only:
      device = torch.device('cuda:%d' % local_rank)
      torch.cuda.set_device(device)

   if rank > 0 and log_level == logging.INFO:
      log_level = logging.WARNING

   logging.basicConfig(level=log_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)


   model_save = args.model_save
   if model_save is None and args.logdir is not None:
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
   logger.info('horovod_groups:     %s',args.horovod_groups)
   logger.info('horovod_data_barrier: %s',args.horovod_data_barrier)
   logger.info('filebase:           %s',args.filebase)
   logger.info('cpu_only:           %s',args.cpu_only)
   logger.info('device:             %s',device)
   logger.info('bf16:               %s',args.bf16)
   logger.info('mixed_precision:    %s',args.mixed_precision)
   logger.info('channels_last:      %s',args.channels_last)
   logger.info('logdir:             %s',args.logdir)

   np.random.seed(args.random_seed)

   config_file = json.load(open(args.config_file))
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   config_file['input_model_pars'] = args.input_model_pars
   config_file['horovod'] = args.horovod
   config_file['horovod_groups'] = args.horovod_groups
   config_file['horovod_data_barrier'] = args.horovod_data_barrier
   config_file['status'] = args.status
   config_file['nval'] = args.nval
   config_file['nval_tests'] = args.nval_tests
   config_file['nsave'] = args.nsave
   config_file['model_save'] = model_save
   config_file['valid_only'] = args.valid_only
   config_file['batch_limiter'] = args.batch_limiter
   config_file['cpu_only'] = args.cpu_only
   config_file['bf16'] = args.bf16
   config_file['channels_last'] = args.channels_last

   if args.filebase is not None:
       config_file['data']['filebase'] = args.filebase

   if args.valid_only and not args.input_model_pars:
      logger.error('if valid_only set, must provide input model')
      return

   # Override config settings from command line if given
   if args.num_parallel_readers >= 0:
      logger.info('Setting the number of parallel readers from command line: %s', args.num_parallel_readers)
      config_file['data']['num_parallel_readers'] = args.num_parallel_readers
   if args.mixed_precision is not None:
      logger.info('Setting NVIDIA auto mixed precision from command line: %s', args.mixed_precision)
      config_file['model']['mixed_precision'] = args.mixed_precision
   if args.train_filelist is not None:
      logger.info('Setting train filelist from command line: %s', args.train_filelist)
      config_file['data']['train_filelist'] = args.train_filelist
   if args.test_filelist is not None:
      logger.info('Setting test filelist from command line: %s', args.test_filelist)
      config_file['data']['test_filelist'] = args.test_filelist
   if args.batch > 0:
      logger.info('setting batch size from command line: %s', args.batch)
      config_file['data']['batch_size'] = args.batch
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
   if config_file['channels_last']:
      net = net.channels_last()

   if rank == 0:
      logger.info('model = \n %s',net)

   total_params = sum(p.numel() for p in net.parameters())

   if rank == 0:
      logger.info('trainable parameters: %s',total_params)

   if args.valid_only:
      valid_model(net,validds,config_file)
   else:
      train_model(net,trainds,testds,config_file,device,writer,args.profile)

def _sync(use_xpu, use_cuda):
  if use_xpu:
     torch.xpu.synchronize()
  if use_cuda:
     torch.cuda.synchronize()
 

def train_model(model,trainds,testds,config,device,writer=None,profile=False):
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
   hvdgroups = config['horovod_groups']
   hvdbarrier = config['horovod_data_barrier']
   num_classes = config['data']['num_classes']
   num_readers = config['data']['num_parallel_readers']
   persistent_workers = False if num_readers == 0 else True

   ## create samplers for these datasets
   train_sampler = torch.utils.data.distributed.DistributedSampler(trainds,nranks,rank,shuffle=True,drop_last=True)
   test_sampler = torch.utils.data.distributed.DistributedSampler(testds,nranks,rank,shuffle=True,drop_last=True)

   ## create data loaders
   train_loader = torch.utils.data.DataLoader(trainds,shuffle=False,
                                            sampler=train_sampler,num_workers=num_readers,
                                            batch_size=batch_size,persistent_workers=persistent_workers)
   test_loader = torch.utils.data.DataLoader(testds,shuffle=False,
                                            sampler=test_sampler,num_workers=num_readers,
                                            batch_size=batch_size,persistent_workers=persistent_workers)

   loss_func = loss.get_loss(config)
   ave_loss = CalcMean.CalcMean()
   acc_func = accuracy.get_accuracy(config)
   ave_acc = CalcMean.CalcMean()
   if not hvdbarrier:
      comm = None
   else:
      from mpi4py import MPI
      comm = MPI.COMM_WORLD

   opt_func = optimizer.get_optimizer(config)
   opt = opt_func(model.parameters(),**config['optimizer']['args'])

   lrsched_func = optimizer.get_learning_rate_scheduler(config)
   lrsched = lrsched_func(opt,**config['lr_schedule']['args'])

   model.to(device)
   if not config['model']['mixed_precision'] and config['bf16']:
      model = model.bfloat16()

   # Add Horovod Distributed Optimizer
   if hvd:
      # for horovod groups
      hvdprms = None
      if hvdgroups is not None:
         hvdsplt = [int(x) for x in hvdgroups.split(",")]
         assert sum(hvdsplt) == 100
         e,b = 0,0
         hvdprms = []
         prms = [v[1] for v in model.named_parameters()][::-1]
         for x in hvdsplt[:-1]:
            e = b + int(len(prms) * x/100)
            hvdprms.append(prms[b:e])
            b = e
         hvdprms.append(prms[b:])
         assert sum([len(x) for x in hvdprms]) == len(prms)
 
      opt = hvd.DistributedOptimizer(opt, named_parameters=model.named_parameters(), groups = hvdprms)
      # Broadcast parameters from rank 0 to all other processes.
      hvd.broadcast_parameters(model.state_dict(), root_rank=0)

   img_secs = []
   use_xpu = True if device.type == "xpu" else False
   use_cuda = True if device.type== "cuda" else False
   mixed_precision = use_cuda and config['model']['mixed_precision']
   bf16 = not mixed_precision and config['bf16']
   for epoch in range(epochs):
      if rank == 0:
         logger.info(' epoch %s of %s',epoch,epochs)
      run_time = 0

      train_sampler.set_epoch(epoch)
      test_sampler.set_epoch(epoch)


      for batch_counter,(inputs,targets,class_weights,nonzero_mask) in enumerate(train_loader):
         # barrier for loading data
         if comm is not None:
            comm.barrier()

         autograd_prof = dummycontext()
         if profile:
            if use_cuda:
               worker_name = 'rank' + str(rank)
               autograd_prof = torch.profiler.profile(activities=[
                       torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA,
                   ], on_trace_ready=torch.profiler.tensorboard_trace_handler('./prof_result', worker_name=worker_name),
                   record_shapes=False,
                   profile_memory=False,
                   with_stack=False
                   )
            elif use_xpu:
               autograd_prof = torch.autograd.profiler_legacy.profile(use_xpu=True)
         with autograd_prof as prof:

            start_time = time.time()
           
            # move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            class_weights = class_weights.to(device)
            nonzero_mask = nonzero_mask.to(device)

            if bf16 is True:
               inputs = inputs.bfloat16()
               targets = targets.bfloat16()
            
            # zero grads
            opt.zero_grad()
            if config['channels_last']:
               inputs = torch.xpu.to_channels_last_1d(inputs)

            # CUDA AMP
            if mixed_precision is True:
               with torch.cuda.amp.autocast():
                  outputs,endpoints = model(inputs)
                  loss_value = loss_func(outputs,targets.long())
                  print(loss_value)
                  # set the weights
               if balanced_loss:
                  weights = class_weights
                  nonzero_to_class_scaler = torch.sum(nonzero_mask.type(torch.float16)) / torch.sum(class_weights.type(torch.float16))
               else:
                  weights = nonzero_mask
                  nonzero_to_class_scaler = torch.ones(1,device=device)

               loss_value = torch.mean(loss_value * weights) * nonzero_to_class_scaler
            else:
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

            # loss acc
            ave_loss.add_value(float(loss_value.to('cpu')))
            # calc acc
            ave_acc.add_value(float(acc_func(outputs,targets,weights).to('cpu')))

            _sync(use_xpu, use_cuda)
            run_time += time.time() - start_time

            # print statistics
            if batch_counter % status == 0:
               rate = config['training']['status'] * batch_size / run_time
               run_time = 0
               img_secs.append(rate)
               logger.info('<[%3d of %3d, %5d of %5d]> train loss: %6.4f acc: %6.4f image/sec: %6.4f',
                   epoch + 1,epochs,batch_counter,len(trainds)/nranks/batch_size,ave_loss.mean(),ave_acc.mean(),rate)

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
               logger.info('batch limiter enabled %5d, stop training early', config['batch_limiter'])
               break

         if profile and prof is not None and use_cuda is False:
            profiling_path = os.environ.get('PROFILE_PATH', '.')
            prof_name = 'pointnet-atlas_rank_' + str(rank) + '_tr_'
            if use_cuda:
               prof_name += 'cuda_'
               sort_key = "self_cuda_time_total"
            elif use_xpu:
               prof_name += 'xpu_'
               sort_key = 'self_xpu_time_total'
            else:
               sort_key = 'self_cpu_time_total'
            prof_name += 'bf16_' if config['bf16'] else 'f32_'
            prof_name += 'chlast_' if config['channels_last'] else 'chfirst_'
            # add epoch info in the end
            prof_name += str(epoch) + '_' + str(batch_counter)
            prof.export_chrome_trace(profiling_path + '/' + prof_name + '.json')
            torch.save(prof.table(sort_by="id", row_limit=100000), profiling_path + '/' + prof_name + '_detailed.pt')
            torch.save(prof.key_averages().table(sort_by=sort_key), profiling_path + '/' + prof_name + '.pt')

      # save at end of epoch
      if writer and rank == 0:
         torch.save(model.state_dict(),model_save + '_%05d.torch_model_state_dict' % epoch)
      
      if nval_tests == -1:
         nval_tests = len(testds)/nranks/batch_size
      logger.info('epoch %s complete, running validation on %s batches',epoch,nval_tests)

      # every epoch, evaluate validation data set
      with torch.no_grad():

         vloss = CalcMean.CalcMean()
         vacc = CalcMean.CalcMean()
         
         vious = [ CalcMean.CalcMean() for i in range(num_classes) ]

         for valid_batch_counter,(inputs,targets,class_weights,nonzero_mask) in enumerate(test_loader):
            
            if config['bf16']:
               inputs = inputs.bfloat16()
               targets = targets.bfloat16()
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

            # calc ious
            ious = get_ious(outputs,targets,weights,num_classes)
            for i in range(num_classes):
               vious[i].add_value(float(ious[i]))

            if config['batch_limiter'] and valid_batch_counter > config['batch_limiter']:
               logger.info('batch limiter enabled %5d, stop validating early', config['batch_limiter'])
               break
            if valid_batch_counter > nval_tests:
               break

         mean_acc = vacc.mean()
         mean_loss = vloss.mean()
         if hvd is not None:
            mean_acc  = hvd.allreduce(torch.tensor([mean_acc]))
            mean_loss = hvd.allreduce(torch.tensor([mean_loss]))
         mious = float(torch.sum(torch.FloatTensor([ x.mean() for x in vious]))) / num_classes
         ious_out = {'jet':vious[0].mean(),'electron':vious[1].mean(),'bkgd':vious[2].mean(),'all':mious}
         # add validation to tensorboard
         if writer and rank == 0:
            global_batch = epoch * len(trainds)/nranks/batch_size + batch_counter
            writer.add_scalars('loss',{'valid':mean_loss},global_batch)
            writer.add_scalars('accuracy',{'valid':mean_acc},global_batch)
            writer.add_scalars('IoU',ious_out,global_batch)
         
         if rank == 0:
            logger.warning('>[%3d of %3d, %5d of %5d]<<< ave valid loss: %6.4f ave valid acc: %6.4f on %s batches >>>',epoch + 1,epochs,batch_counter,len(trainds)/nranks/batch_size,mean_loss,mean_acc,valid_batch_counter + 1)
            logger.warning('      >> ious: %s',ious_out)
         

      # update learning rate
      lrsched.step()        

   warm_up = 4 if len(img_secs) > 5 else 0
   if (warm_up == 0):
       print('Warning: no warm up, performance data might not be solid...')
   img_sec_mean = np.mean(img_secs[warm_up:])
   if hvd:
       logger.info('avg imgs/sec on rank %d: %.2f' % (rank, img_sec_mean))
       if rank == 0:
          logger.info('total imgs/sec on %d ranks: %.2f' % (nranks, nranks * img_sec_mean))
   else:
       logger.info('avg imgs/sec: %.2f' % (img_sec_mean))
       logger.info('total imgs/sec: %.2f' % (img_sec_mean))


def get_ious(pred,labels,weights,num_classes,smooth=1,point_axis=1):
   # Implicit dimension choice for softmax has been deprecated
   pred = torch.nn.functional.softmax(pred, dim=pred.dim()-1)
   pred = pred.argmax(dim=point_axis)
   
   ious = []
   for i in range(num_classes):
      class_pred = (pred == i).int() * weights
      class_label = (labels == i).int() * weights
      intersection = torch.sum(class_label * class_pred,dim=point_axis)
      union = torch.sum(class_label,dim=point_axis) + torch.sum(class_pred,dim=point_axis) - intersection
      iou = torch.mean( (intersection + smooth) / (union + smooth), dim=0 )

      ious.append(iou)

   return ious

if __name__ == "__main__":
   main()
