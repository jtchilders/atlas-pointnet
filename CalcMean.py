import numpy as np


class CalcMean:
   def __init__(self):
      self.reset()

   def reset(self):
      self.smean = np.zeros(1)
      self.sigma = np.zeros(1)
      self.n = np.zeros(1)
      self.sum = np.zeros(1)
      self.sum2 = np.zeros(1)

   def add_value(self,value):
      self.n += 1
      self.sum += value
      self.sum2 += value * value
      self.smean = np.zeros(1)
      self.sigma = np.zeros(1)

   def mean(self):
      if self.smean != 0:
         return self.smean
      if self.n == 0:
         return 0

      self.smean = float(self.sum) / float(self.n)
      return self.smean

   def std(self):
      if self.sigma != 0:
         return self.sigma
      if self.n == 0:
         return 0
      mean = self.mean()
      self.sigma = np.sqrt((1. / self.n) * self.sum2 - mean * mean)
      return self.sigma

   def allreduce(self,hvd):
      self.n = hvd.allreduce(np.array([self.n]),op=hvd.mpi_ops.Sum)
      self.sum = hvd.allreduce(np.array([self.sum]),op=hvd.mpi_ops.Sum)
      self.sum2 = hvd.allreduce(np.array([self.sum2]),op=hvd.mpi_ops.Sum)
      self.smean = np.zeros(1)
      self.sigma = np.zeros(1)

   def __add__(self,other):
      new = CalcMean()
      new.n = self.n + other.n
      new.sum = self.sum + other.sum
      new.sum2 = self.sum2 + other.sum2
      return new

   def __eq__(self,other):
      if self.mean() == other.mean():
         return True
      return False

   def __ne__(self,other):
      if self.mean() != other.mean():
         return True
      return False

   def __gt__(self,other):
      if self.mean() > other.mean():
         return True
      return False

   def __lt__(self,other):
      if self.mean() < other.mean():
         return True
      return False

   def __ge__(self,other):
      if self.mean() >= other.mean():
         return True
      return False

   def __le__(self,other):
      if self.mean() <= other.mean():
         return True
      return False

   def get_string(self,format='%f +/- %f',
                  show_percent_error=False,
                  show_percent_error_format='%12.2f +/- %12.2f (%04.2f%%)'):
      s = ''
      if show_percent_error:
         percent_error = self.calc_sigma() / self.calc_mean() * 100.
         s = show_percent_error_format % (self.calc_mean(),self.calc_sigma(),percent_error)
      else:
         s = format % (self.calc_mean(),self.calc_sigma())
      return s


class FifoMean:
   def __init__(self,fifo_size=10):
      self.fifo_size = fifo_size
      self.reset()

   def reset(self):
      self.fifo = np.zeros(self.fifo_size)
      self.counter = 0

   def add_value(self,value):
      self.fifo[self.counter] = value
      self.increment_counter()

   def increment_counter(self):
      self.counter += 1
      if self.counter >= self.fifo_size:
         self.counter = 0

   def mean(self):
      return self.fifo.mean()

   def std(self):
      return self.fifo.std()
