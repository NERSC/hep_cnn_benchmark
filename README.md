# High Energy Physics Deep Learning Convolutional Neural Network Benchmark
TensorFlow Benchmark for the HEP Deep Learning Model used in arXiv:1708.05256 which uses 224x224x3 images and a smaller variant using 64x64x3 images.

*** Copyright Notice ***
High Energy Physics Deep Learning Convolutional Neural Network Benchmark (HEPCNNB) Copyright (c) 2017, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Innovation and Partnerships Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit other to do so.
****************************

## Code Structure
The main python training scripts are in `scripts`. These scripts do the command
line parsing, set up distributed training, and call the training loop.
- `hep_classifier_tf_train.py` is for standard Tensorflow serial and distributed training.
- `hep_classifier_tf_train_horovod.py` is for distributed Tensorflow training with Horovod.
- `hep_classifier_tf_train_craype-ml.py` is for distributed Tensorflow training
   with the Cray ML plugin.

The CNN classifier model is defined in `networks/binary_classifier_tf.py` along with
some data utility code and routines for model performance metrics.

The `slurm_tf_helper/setup_clusters.py` module contains some helper code which
parses SLURM parameters and returns a cluster object which can be used by
TensorFlow for distributed training.

The `configs` folder holds json configuration files for running distributed training on the NERSC Cori system, either on the Xeon Phi 7250 (Knight's Landing, KNL) partition or the Intel Xeon Haswell partition. These files further contain the network parameters for the training and the paths to the relevant data. Users need to specify the `inputpath` and point it to their data directory. The data description is given [below](## Data description).

The folder `run_scripts` contains batch scripts for running distributed training on the NERSC Cori machine. They set environment parameters, correct thread binding and encapsulate other boiler-plate settings so that the user does not need to worry about that. They are supposed to be submitted from that directory, otherwise the relative path logic will break. By default, the scripts submit the 224x224x3 image size case but they can also be used for submitting the smaller network. This can be done by changing the `--config` argument to point to the corresponding json file in the previously mentioned `configs` folder.
In order to submit these scripts at NERSC, do

    sbatch -N <numnodes> run_scripts/run_cori_knl.sh

for running distributed training on `<numnodes>` nodes on the Cori KNL partition.
If `-N` is not specified, the training will be performed on a single node.
By default, the distributed training uses 1 parameters server if the number of nodes is bigger than one. However, this can easily be changed by using the `--num_ps` variable in the run scripts.

## Data description
The data represents simulated detector data from a (non-existant) general purpose detector which shares many features with [ATLAS detector at CERN](http://atlas.cern). The dataset was generated with [Delphes](https://arxiv.org/pdf/1307.6346.pdf), a fast Monte-Carlo generator for particle collisions. This simulator computes particle collisions and emulates a parton shower, i.e. particles produced in the collision propagate and eventually decay or get absorbed by the detector material. This whole process is called an *event*. The original data consisted of cylindrical coordinates with energy deopisits in the detectors Hadron Calorimeter as well as Electromagnetic Calorimeter as well as track information inferred from the pixel detector. This continuous data is binned to 64x64x3 (small network) or 224x224x3 (large network) size images. The x-dimension represents the azimuth angle $\phi$ and the y-dimension the pseudo-rapidity $\eta$. Thus the images represent an unrolled 2D cylindrical detector topology. The three channels are: total energy deposit (Hadron + EM calorimeter), energy fraction deposited in EM calorimeter, track multiplicity.
Other channels could be used and other image sizes at well but that is up to the user of this benchmark. All the data is normalized per channel by dividing out the maxium value over the training sample and image pixels.

In our test, we generated particle events which can be fully described by standard model physics and particle events which contain new physics, in our case R-parity violating Supersymmetry (check [this link](https://link.springer.com/article/10.1007/BF02908093) for a review). However, the user is feel to generate his own data with different event types. The model can also be easily expanded to perform a multi-class classification or even include a regression on model parameters.

For NERSC users, we can give access to the our dataset on request. Please send an email to (consult@nersc.gov) and ask for acces to the HEP TensorFlow CNN benchmark data.
For users who want to generate their own data, this is the data format which is expected by the data iterator specified in `networks/binary_classifier_tf.py`.
The users are free to edit this file or stick to our convention which is:

    h5ls <trainfile-name>.h5
    data                     Dataset {<nsamples>, 3, 224, 224}
    label                    Dataset {<nsamples>}
    normweight               Dataset {<nsamples>}
    weight                   Dataset {<nsamples>}

We will give a short decription of the data here:

`<nsamples>`: number of samples in the file
`data` : images which are fed to the neural network
`label`: integer label
`normweight`: normalized weight. Each sample is weighted by that weight given here. Currently overridden and set to 1, but can be changed. The normalization should be done over the training set.
`weight`: unnormalized weight. That field can be used to compute more sophisticated performance metrics to account for highly skewed data sets.

## Dummy data
The benchmark can be run with dummy data only. For that purpose, just append `--dummy_data` to the command line arguments of the main python script. The loss is meaningless as random data will be generated for the training process but this mode can be used to perform IO insensitive scaling tests.
