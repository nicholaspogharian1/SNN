#!/bin/bash
#SBATCH -p buyin
#SBATCH -A b1164				# job allocation
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --ntasks-per-node=24     # Number of Cores (Processors)

################################
#THIS SCRIPT RUNS THE LANG CODE#
################################

#COMMAND LINE ARGS
# $1 - number of neurons in the network
# $2 - theta value for the neurons

#import appropriate python modules and such (quick and dirty for now)
module purge
module load python-miniconda3
source activate SNN-env

#enter the appropriate directory
cd /projects/b1021/Nick/SNN

python SNN.py $1 $2
