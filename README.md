# atlas_dl_benchmark
TensorFlow Benchmark for the HEP Deep Learning Model used in arXiv:1708.05256 which uses 224x224x3 images and a smaller variant using 64x64x3 images.

## Code Structure

The main python driver routine is ```hep_classifier_tf_train.ipynb```. This file does the argument parsing as well as setting up the distributed training and calling the training loop. 
This notebook includes some standard packages as well as the ```slurm_tf_helper module``` and the network file ```binary_classifier_tf.ipynb``` located in ```networks```. The latter contains the network setup routines as well as the setup routines for performance metrics. 

The slurm helper module is a small class which parses SLURM parameters and returns a cluster object which can be used by TensorfFlow for distributed training.

The file ```make_scripts.sh``` in the main directory transforms the Jupyter notebooks into python scripts so that they can be used in a batch session. The scripts will be placed in the subdirectory ```scripts``` where the directory hierarchy of the original jupyter notebook files is recpliated. 

The folder ```configs``` holds json configuration files for running distributed training on the NERSC Cori system, either on the Xeon Phi 7250 (Knight's Landing, KNL) partition or the Intel Xeon Haswell partition. These files further contain the network parameter for the training and the paths to the relevant data. Users need to specify the ```inputpath``` and point it to their data directory. The data description is given [below](## Data description).

The folder ```run_scripts``` contains batch scripts for running distributed training on the NERSC Cori machine. They set environment parameters, correct thread binding and encapsulate other boiler-plate settings so that the user does not need to worry about that. They are supposed to be submitted from that directory, otherwise the relative path logic will break. By default, the scripts submit the 224x224x3 image size case but they can also be used for submitting the smaller network. This can be done by changing the ```--config``` argument to point to the corresponding json file in the previously mentioned ```configs``` folder. 
In order to submit these scripts at NERSC, do

```
sbatch -N <numnodes> run_cori_knl.sh
```

for running distributed training on ```<numnodes>``` nodes on the Cori KNL partition. If ```-N``` is not specified, the training will be performed on a single node.
By default, the distributed training uses 1 parameters server if the number of nodes is bigger than one. However, this can easily be changed by using the ```--num_ps``` variable in the run scripts.

## Data description
The data represents 

