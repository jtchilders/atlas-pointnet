# atlas-pointnet
applying pointnet++ to ATLAS data.

Running on JLSE 4 V100s using
``` bash
mpirun -hostfile $COBALT_NODEFILE -n 4 python run_pointnet.py -c configs/jlse.json --logdir logdir/$(date "+%Y-%m-%d")/$TRIAL --horovod
```
After 30 epochs, reached 60% accuracy on training data, 58% on testing data.
Image throughput is about 60 images/second/gpu.

