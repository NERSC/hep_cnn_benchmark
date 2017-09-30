#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 8:00:00
#SBATCH -J hep_train_tf

#set up python stuff
module load python
source activate thorstendl-devel
export PYTHONPATH=/cori_knl_small.json/usr/common/software/tensorflow/intel-tensorflow/head/lib/python2.7/site-packages
#export PYTHONPATH=/global/homes/t/tkurth/python/tfzoo/tensorflow_mkl_hdf5_mpi_cw

#run
cd ../scripts/
python hep_classifier_tf_train.py --config=../configs/cori_knl_small.json --num_tasks=1 --num_ps=0
