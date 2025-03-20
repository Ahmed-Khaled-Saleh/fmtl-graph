#!/bin/bash
#SBATCH --account=project_2009050
#SBATCH --job-name=dmtl
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./logs/out.log
#SBATCH --error=./logs/err.log

module --force purge
module load pytorch
source /projappl/project_2009050/mytorch/bin/activate
pip uninstall -y fedai
pip install git+https://github.com/Ahmed-Khaled-Saleh/fedai.git
cd /projappl/project_2009050/fmtl-graph

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/mytorch/lib/python3.11/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"

srun python main.py --config ./cfgs/cfg.yaml --env_file ./.env --lr2 0.00002
