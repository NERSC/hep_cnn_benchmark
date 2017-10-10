#!/bin/bash
#SBATCH -p regular
#SBATCH -A nstaff
#SBATCH -C haswell
#SBATCH -t 1:00:00
#SBATCH -J hep_train_tf


#BSD 3-Clause License
#
#Copyright (c) 2017, The Regents of the University of California, 
#through Lawrence Berkeley National Laboratory 
#(subject to receipt of any required approvals from the U.S. Dept. of Energy)
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#* Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#* Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#* Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------


#set up python stuff
module load python
source activate thorstendl-devel
export PYTHONPATH=/usr/common/software/tensorflow/intel-tensorflow/head/lib/python2.7/site-packages

#run
cd ../scripts/

if [ $SLURM_NNODES -gt 2 ]; then
    NUM_PS=1
else
    NUM_PS=0
fi

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 64 -u python hep_classifier_tf_train.py --config=../configs/cori_haswell_224.json --num_tasks=${SLURM_NNODES} --num_ps=${NUM_PS}
