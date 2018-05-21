#!/bin/bash
#SBATCH -q regular
#SBATCH -C knl,quad,cache
#SBATCH -t 4:00:00
#SBATCH --gres=craynetwork:2
#SBATCH -J hep_train_tf


#*** License Agreement ***
#
#High Energy Physics Deep Learning Convolutional Neural Network Benchmark (HEPCNNB) Copyright (c) 2017, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#(1) Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#(2) Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#(3) Neither the name of the University of California, Lawrence Berkeley National Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades to the features, functionality or performance of the source code ("Enhancements") to anyone; however, if you choose to make your Enhancements available either publicly, or directly to Lawrence Berkeley National Laboratory, without imposing a separate written license agreement for such Enhancements, then you hereby grant the following license: a  non-exclusive, royalty-free perpetual license to install, use, modify, prepare derivative works, incorporate into other computer software, distribute, and sublicense such enhancements or derivative works thereof, in binary and source code form.
#---------------------------------------------------------------

#set up python stuff
module load tensorflow/intel-horovod-mlsl-1.6
export PYTHONPATH=/global/homes/t/tkurth/.conda/envs/helper-env-py2/lib/python2.7/site-packages:${PYTHONPATH}

#MLSL stuff
export MLSL_NUM_SERVERS=2
export EPLIB_MAX_EP_PER_TASK=${MLSL_NUM_SERVERS}
export EPLIB_UUID="00FF00FF-0000-0000-0000-00FF00FF00FF"
export EPLIB_DYNAMIC_SERVER="disable"
export EPLIB_SERVER_AFFINITY=67,66
#export MLSL_LOG_LEVEL=5
export EPLIB_SHM_SIZE_GB=20
export MLSL_SHM_SIZE_GB=20
#export TF_MKL_ALLOC_MAX_BYTES=$((16*1024*1024*1024))
export USE_HVD=1
export PYTHONUNBUFFERED=1
export USE_MLSL_ALLOCATOR=1
export EP_PROCESS_NUM=$((${SLURM_NNODES}*${MLSL_NUM_SERVERS}))

#better binding
bindstring="numactl -C 1-67,69-135,137-203,205-271"

#run
cd ../scripts/

#launch MLSL server
srun -N ${SLURM_NNODES} -n ${EP_PROCESS_NUM} -c 4 --mem=37200 --gres=craynetwork:1 ${MLSL_ROOT}/intel64/bin/ep_server &

#launch srun
srun -l --zonesort=off -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 264 --mem=37200 --gres=craynetwork:0 -u ${bindstring} ./run_mlsl_wrapper.sh
wait
