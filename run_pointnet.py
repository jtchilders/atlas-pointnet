#!/usr/bin/env python3
import argparse,logging,socket,json,sys
import numpy as np
from data_handlers import utils as datautils
import torch
import tensorboardX

logger = logging.getLogger(__name__)



def main():
   ''' simple starter program that can be copied for use when starting a new script. '''

   parser = argparse.ArgumentParser(description='')
   parser.add_argument('-c','--config_file',help='configuration file in json format',required=True)
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--model_save',default='model_saves',help='base name of saved model parameters for later loading')
   parser.add_argument('--nsave',default=100,type=int,help='frequency in batch number to save model')

   parser.add_argument('--nval',default=100,type=int,help='frequency to evaluate validation sample in batch numbers')
   parser.add_argument('--nval_tests',default=1,type=int,help='number batches to test per validation run')

   parser.add_argument('--status',default=20,type=int,help='frequency to print loss status in batch numbers')

   parser.add_argument('--batch',default=-1,type=int,help='set batch size, overrides file config')

   parser.add_argument('--random_seed',default=0,type=int,help='numpy random seed')

   parser.add_argument('--valid_only',default=False,action='store_true',help='flag that triggers validation run. prints confusion matrix.')

   parser.add_argument('--batch_limiter',help='if set to an integer, will limit the number of batches during training. Use this to create short training runs for profiling.',type=int)

   parser.add_argument('-i','--input_model_pars',help='if provided, the file will be used to fill the models state dict from a previous run.')
   parser.add_argument('-e','--epochs',type=int,default=-1,help='number of epochs')
   parser.add_argument('-l','--logdir',help='log directory for tensorboardx')

   parser.add_argument('--horovod',default=False, action='store_true', help="Setup for distributed training")

   parser.add_argument('--filelist_base',default='filelist',help='base filename for the output filelists')

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
   hvd = None
   if args.horovod:
      print('importing horovod')
      import horovod.torch as hvd
      print('imported horovod')
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      logging_format = '%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s'

   if rank > 0 and log_level == logging.INFO:
      log_level = logging.WARNING

   logging.basicConfig(level=log_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   logger.info('rank %s of %s',rank,nranks)
   logger.info('hostname:           %s',socket.gethostname())
   logger.info('python version:     %s',sys.version)

   logger.info('config file:        %s',args.config_file)
   logger.info('num files:          %s',args.num_files)
   logger.info('model_save:         %s',args.model_save)
   logger.info('random_seed:        %s',args.random_seed)
   logger.info('valid_only:         %s',args.valid_only)
   logger.info('nsave:              %s',args.nsave)
   logger.info('nval:               %s',args.nval)
   logger.info('nval_tests:         %s',args.nval_tests)
   logger.info('status:             %s',args.status)
   logger.info('input_model_pars:   %s',args.input_model_pars)
   logger.info('epochs:             %s',args.epochs)
   logger.info('horovod:            %s',args.horovod)
   logger.info('logdir:             %s',args.logdir)
   logger.info('batch_limiter:      %s',args.batch_limiter)
   logger.info('num_threads:        %s',torch.get_num_threads())
   logger.info('filelist_base:      %s',args.filelist_base)

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
   config_file['model_save'] = args.model_save
   config_file['valid_only'] = args.valid_only
   config_file['batch_limiter'] = args.batch_limiter
   config_file['filelist_base'] = args.filelist_base

   if args.valid_only and not args.input_model_pars:
      logger.error('if valid_only set, must provide input model')
      return

   if args.batch > 0:
      config_file['training']['batch_size'] = args.batch
   if args.epochs > 0:
      config_file['training']['epochs'] = args.epochs

   logger.info('configuration = \n%s',json.dumps(config_file, indent=4, sort_keys=True))
   config_file['hvd'] = hvd

   # get datasets for training and validation
   trainds,validds = datautils.get_datasets(config_file)
   
   # setup tensorboard
   writer = None
   if args.logdir and rank == 0:
      writer = tensorboardX.SummaryWriter(args.logdir)
   
   logger.info('building model')
   if 'pytorch' in config_file['model']['framework']:
      from pytorch import model,loss

      net = model.get_model(config_file)
      if hasattr(trainds,'dataset') and hasattr(net,'output_grid'):
         trainds.dataset.grid_size = net.output_grid

      if hasattr(validds,'dataset') and hasattr(net,'output_grid'):
         validds.dataset.grid_size = net.output_grid

      opt,lrsched = model.setup(net,hvd,config_file)

      #lossfunc = loss.get_loss(config_file)
      #accfunc = loss.get_accuracy(config_file)

      logger.info('model = \n %s',net)

      total_params = sum(p.numel() for p in net.parameters())
      logger.info('trainable parameters: %s',total_params)

      if args.valid_only:
         model.valid_model(validds,config_file)
      else:
         model.train_model(net,opt,lrsched,trainds,validds,config_file,writer)
            

def print_module(module,input_shape,input_channels,name=None,indent=0):

   output_string = ''
   output_channels = input_channels
   output_shape = input_shape

   output_string += '%10s' % ('>' * indent)
   if name:
      output_string += ' %20s' % name
   else:
      output_string += ' %20s' % module.__class__.__name__

   # convolutions change channels
   if 'submanifoldconv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.nIn,module.nOut)
      output_channels = module.nOut
   elif 'conv' in module.__class__.__name__.lower():
      output_string += ' %4d -> %4d ' % (module.in_channels,module.out_channels)
      output_channels = module.out_channels
   elif 'pool' in module.__class__.__name__.lower():
      output_shape = [int(input_shape[i] / module.pool_size[i]) for i in range(len(input_shape))]
      output_string += ' %10s -> %10s ' % (input_shape, output_shape)
   elif 'batchnormleakyrelu' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.nPlanes
   elif 'batchnorm2d' in module.__class__.__name__.lower():
      output_string += ' (%10s) ' % module.num_features

   output_string += '\n'

   for name, child in module.named_children():
      string,output_shape,output_channels = print_module(child, output_shape, output_channels, name, indent + 1)
      output_string += string

   return output_string,output_shape,output_channels


def summary(input_shape,input_channels,model):

   return print_module(model,input_shape,input_channels)


if __name__ == "__main__":
   main()
