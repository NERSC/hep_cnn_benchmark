#!/bin/bash
#SBATCH -J hep_cnn_train
#SBATCH -t 00:30:00
#SBATCH -p normal
#SBATCH -C gpu


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

#module load
module unload PrgEnv-cray
module load PrgEnv-gnu
module load gcc/5.3.0
module load cudatoolkit/8.0.61_2.4.3-6.0.4.0_3.1__gb475d12
source activate tensorflow
module load /scratch/snx3000/pjm/tmp_inst/modulefiles/craype-ml-plugin-py2/1.1.0

#openmp stuff
export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export CRAY_CUDA_MPS=1
export MPICH_RDMA_ENABLED_CUDA=1

#run
cd ../scripts/

srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} --cpu_bind=rank -u python hep_classifier_tf_train_craype-ml.py --config=../configs/daint_gpu_224.json --num_tasks=${SLURM_NNODES} > hep_224x224_p100-craype-ml_w$(( ${SLURM_NNODES} ))_p0.out 2>&1
