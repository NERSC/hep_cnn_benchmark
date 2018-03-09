#!/bin/bash

runscript="./run_cori_knl_craype-ml.sh"

jobid=$(sbatch -N 1024 ${runscript} | awk '{print $4}')
for n in 512 256 128 64 32 16 8 4 2 1; do
    submitstring="sbatch -d afterany:${jobid} -N ${n} ${runscript}"
    echo ${submitstring}
    jobid=$(${submitstring} | awk '{print $4}')
done
