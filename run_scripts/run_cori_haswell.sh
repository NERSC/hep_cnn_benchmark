#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -C haswell
#SBATCH -t 1:00:00
#SBATCH -J hep_train_tf

#set up python stuff
module load python
source activate thorstendl-devel
export PYTHONPATH=/cori_knl_small.json/usr/common/software/tensorflow/intel-tensorflow/head/lib/python2.7/site-packages

#run
cd ../scripts/

if [ $SLURM_NNODES -gt 2 ]; then
    NUM_PS=1
else
    NUM_PS=0
fi

python hep_classifier_tf_train.py --config=../configs/cori_haswell_224.json --num_tasks=${SLURM_NNODES} --num_ps=${NUM_PS}
