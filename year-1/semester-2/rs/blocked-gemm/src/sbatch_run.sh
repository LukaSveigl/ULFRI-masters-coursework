#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=1 
#SBATCH --output=test2.out
#SBATCH --constraint=amd
#SBATCH --reservation=fri
#SBATCH --time=04:00:00

FILE=hpc_gemm


# 01 - no optimization 
gcc  ${FILE}.c -g -O1 -Wall -o ${FILE}
#gcc -S ${FILE}.c 

# ./${FILE}
srun --job-name=test --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --output=test2.out --constraint=amd --reservation=fri --time=04:00:00 ${FILE}
rm ./${FILE}
