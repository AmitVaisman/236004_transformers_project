#!/bin/bash
#
CONDA_HOME=~/miniconda3
CONDA_ENV=236004_transformers_project

unset XDG_RUNTIME_DIR
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# Check if the environment is activated
#echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
#echo "Python path: $(which python)"
#echo "Jupyter path: $(which jupyter)"

# Run Jupyter
/home/amitf/.conda/envs/subnetvenv/bin/jupyter --notebook-dir=./ --no-browser \
 --ip=$(hostname -I | cut -d " " -f1) --port-retries=100
