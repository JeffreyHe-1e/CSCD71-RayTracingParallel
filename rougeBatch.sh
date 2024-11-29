#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

module load rocm/4.0.1
module load gcc/11.1.0

