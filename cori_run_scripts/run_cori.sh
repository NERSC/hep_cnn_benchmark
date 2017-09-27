#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -N 9
#SBATCH -C haswell
#SBATCH -t 8:00:00
#SBATCH -J hep_train_tf

#set up python stuff
module load python/3.6-anaconda-4.4
module load h5py
module load tensorflow/intel-head

#run
python ../scripts/hep_classifier_tf_train.py
