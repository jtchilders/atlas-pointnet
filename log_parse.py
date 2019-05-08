#!/usr/bin/env python3
import argparse,logging,json,time,subprocess
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)


def main():
   ''' simple starter program that can be copied for use when starting a new script. '''
   logging_format = '%(asctime)s %(levelname)s:%(name)s:%(message)s'
   logging_datefmt = '%Y-%m-%d %H:%M:%S'
   logging_level = logging.INFO
   
   parser = argparse.ArgumentParser(description='parse log file for loss and timing information throughout run')
   parser.add_argument('-i','--input',dest='input',help='log file produced by atlas-identify-pytorch-sparse.',required=True)
   parser.add_argument('-o','--output',dest='output',help='output json file')
   parser.add_argument('-p','--outputfig',dest='outputfig',help='output figure name.')
   parser.add_argument('-t','--sleep',dest='sleep',help='time between parsing.',default=10,type=int)
   parser.add_argument('-r','--repeat',dest='repeat',help='number of times to repeat',default=1000,type=int)

   parser.add_argument('--debug', dest='debug', default=False, action='store_true', help="Set Logger to DEBUG")
   parser.add_argument('--error', dest='error', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--warning', dest='warning', default=False, action='store_true', help="Set Logger to ERROR")
   parser.add_argument('--logfilename',dest='logfilename',default=None,help='if set, logging information will go to file')
   args = parser.parse_args()

   if args.debug and not args.error and not args.warning:
      logging_level = logging.DEBUG
   elif not args.debug and args.error and not args.warning:
      logging_level = logging.ERROR
   elif not args.debug and not args.error and args.warning:
      logging_level = logging.WARNING

   logging.basicConfig(level=logging_level,
                       format=logging_format,
                       datefmt=logging_datefmt,
                       filename=args.logfilename)

   if args.output is None:
      args.output = args.input + '.json'
   if args.outputfig is None:
      args.outputfig = args.input + '.png'

   logger.info('parsing log file:   %s',args.input)
   logger.info('output json file:   %s',args.output)
   logger.info('output image file:  %s',args.outputfig)
   logger.info('sleep seconds:      %s',args.sleep)

   for _ in range(args.repeat):
      logger.info('parsing data')
      data = parse_file(args.input)

      json.dump(data,open(args.output,'w'),indent=4, sort_keys=True)

      plot_data(data,args.outputfig)

      logger.info('sleeping: %s',args.sleep)
      time.sleep(args.sleep)


def parse_file(filename):

   # get train data
   training_out,training_err = grep('"<\[.*of.*of.*\]>"',filename)

   # get valid data
   valid_out,valid_err = grep('">\[.*of.*of.*\]<"',filename)

   # get rank info
   rank_out,rank_err = grep('":rank.*of"',filename)
   rank = int(rank_out[rank_out.find(':rank') + 5:].strip().split()[0])
   nranks = int(rank_out[rank_out.find(':rank') + 5:].split()[2])

   # get batch size
   bs_out,bs_err = grep('"\\"batch_size\\":"',filename)
   batch_size = int(bs_out.strip().split()[1][:-1])

   batch_vs_loss = []
   batch_vs_acc = []

   valid_batch_vs_loss = []
   valid_batch_vs_acc = []

   training_data = []
   valid_data = []
   batch_size = None

   for line in training_out.split('\n'):
      if len(line) == 0: continue
      data = {}
      epoch,nepochs,batch,nbatches = get_line_header(line)
      data['epoch'] = epoch
      data['nepochs'] = nepochs
      data['batch'] = batch
      data['nbatches'] = nbatches
      data['loss'] = float(get_value(line,'train loss:'))
      data['acc'] = float(get_value(line,'train acc:'))
      data['imgs_sec'] = float(get_value(line,'images/sec:'))
      data['data_time'] = float(get_value(line,'data time:'))
      data['fwd_time'] = float(get_value(line,'forward time:'))
      data['bwd_time'] = float(get_value(line,'backward time:'))

      data['step'] = (data['epoch'] - 1) * data['nbatches'] + (data['batch'] - 1)

      batch_vs_loss.append([data['step'],data['loss']])
      batch_vs_acc.append([data['step'],data['acc']])

      training_data.append(data)

   for line in valid_out.split('\n'):
      if len(line) == 0: continue
      
      data = {}
      epoch,nepochs,batch,nbatches = get_line_header(line,header_start='>[',header_end=']<')
      data['epoch'] = epoch
      data['nepochs'] = nepochs
      data['batch'] = batch
      data['nbatches'] = nbatches
      data['step'] = (data['epoch'] - 1) * data['nbatches'] + (data['batch'] - 1)
      data['loss'] = float(get_value(line,'valid loss:'))
      data['acc'] = float(get_value(line,'valid acc:'))

      valid_batch_vs_loss.append([data['step'],data['loss']])
      valid_batch_vs_acc.append([data['step'],data['acc']])

      valid_data.append(data)

   output = {'training':training_data,
             'valid':valid_data,
             'train_loss':batch_vs_loss,
             'train_acc':batch_vs_acc,
             'valid_loss':valid_batch_vs_loss,
             'valid_acc':valid_batch_vs_acc,
             'batch_size': batch_size,
             'nranks': nranks,
             'rank': rank,
            }

   logger.info('entries: %s',len(valid_batch_vs_loss))
   
   return output


def get_value(line,search):
   start_index = line.find(search) + len(search)
   return line[start_index:].strip().split()[0]


def get_line_header(line,header_start='<[',header_end=']>'):
   start_index = line.find(header_start) + len(header_start)
   end_index = line.find(header_end)
   header = line[start_index:end_index]
   parts = header.split()
   epoch = int(parts[0])
   nepochs = int(parts[2][:-1])
   batch = int(parts[3])
   nbatches = int(parts[5])

   return epoch,nepochs,batch,nbatches


def grep(string,filename):
   # get rank info
   p = subprocess.Popen('grep %s %s' % (string,filename),shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   out,err = p.communicate()
   return out.decode("utf-8"),err.decode("utf-8")


def plot_data(data,outputfig):

   train_loss = np.array(data['train_loss'])
   train_acc = np.array(data['train_acc'])

   valid_loss = np.array(data['valid_loss'])
   valid_acc = np.array(data['valid_acc'])

   fig,(ax1,ax2) = plt.subplots(2,figsize=(15,15),dpi=80)

   ax1.plot(train_loss[...,0],train_loss[...,1],label='train loss')
   ax1.plot(valid_loss[...,0],valid_loss[...,1],label='valid loss')
   ax1.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax1.grid(axis='y')
   # ax1.set_ylim([0,2])
   # ax1.set_yscale('log')

   ax2.plot(train_acc[...,0],train_acc[...,1],label='train acc')
   ax2.plot(valid_acc[...,0],valid_acc[...,1],label='valid acc')
   ax2.legend(loc='upper center', shadow=False, fontsize='x-large')
   ax2.grid(axis='y')
   # ax2.set_ylim([0,2])
   # ax2.set_yscale('log')

   # logger.info('\n %s \n %s',dir(ax4),dir(fig))
   # logger.info('\n %s \n %s',ax4.lines,dir(ax4.lines[0]))

   plt.savefig(outputfig)


if __name__ == "__main__":
   main()
