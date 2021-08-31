# atlas-pointnet
applying pointnet++ to ATLAS data.

Running on JLSE 4 V100s using
``` bash
mpirun -hostfile $COBALT_NODEFILE -n 4 python run_pointnet.py -c configs/jlse.json --logdir logdir/$(date "+%Y-%m-%d")/$TRIAL --horovod
```
After 30 epochs, reached 60% accuracy on training data, 58% on testing data.
Image throughput is about 60 images/second/gpu.

The dataset used is found here:

https://opendata-qa.cern.ch/record/15009


After the file is unpacked you can generate the training/validation file lists via:

```bash
ls -d /path/to/zej/* | head -n 80000 > training_filelist.txt
ls -d /path/to/zej/* | tail -n 20000 > validation_filelist.txt
```

Then update the `config/jlse.json` config file to use these filelists instead of the existing ones.

