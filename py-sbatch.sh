#!/bin/bash

###
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=2
NUM_CORES=2
NUM_GPUS=4
JOB_NAME="transformers"
MAIL_USER="amit.vaisman@campus.technion.ac.il"
MAIL_TYPE=END # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=236004_transformers_project


sbatch \
  -N $NUM_NODES \
  -c $NUM_CORES \
  -A tamar \
  -p tamar \
  --gres=gpu:$NUM_GPUS \
  --job-name $JOB_NAME \
  --mail-user $MAIL_USER \
  --mail-type $MAIL_TYPE \
  -o 'slurm/%j-%N.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Define the lists
# Run python with the args to the script
python -u $@

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF



