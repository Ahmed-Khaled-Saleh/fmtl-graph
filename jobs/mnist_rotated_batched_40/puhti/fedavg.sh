#!/bin/bash
ts=$(date +%Y%m%d_%H%M%S)
#SBATCH --account=project_2009050
#SBATCH --job-name=fedavg
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./logs/out_${ts}_%j_%x_%N.log  # includes time stamp (t), job ID(j), job name (x), and node name (N)
#SBATCH --error=./logs/err_${ts}_%j_%x_%N.log

module --force purge
module load pytorch
source /projappl/project_2009050/code/FMTL-graph/mytorch/bin/activate
pip uninstall -y fedai
pip install git+https://github.com/Ahmed-Khaled-Saleh/fedai.git
cd /projappl/project_2009050/fmtl-graph

export PYTHONPATH=$PYTHONPATH:/projappl/project_2009050/code/FMTL-graph/mytorch/lib/python3.11/site-packages
echo "Current PYTHONPATH: $PYTHONPATH"

srun python main.py --config ./cfgs/mnist_rotated_batched_40/fedavg.yaml --env_file ./.env --timestamp ${ts}
