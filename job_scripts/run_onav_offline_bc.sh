#!/bin/bash
#SBATCH --job-name=onav_bc
#SBATCH -p gpu_p13               # Name of the partition 
#SBATCH -C v100-32g 
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # Number of GPUs on a node
#SBATCH --gres gpu:1             # The same
#SBATCH -c 10                    # Number of workers per GPU
#SBATCH --hint=nomultithread     # Logical cores
#SBATCH --time 50:00:00          # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/%j.out # Output file name
#SBATCH --error=slurm_logs/%j.out  # Error file name

cd /home/shichen/codes/onav_rim

export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

python_bin=$HOME/miniconda3/envs/onav/bin/python

# configfile=offline_bc/config/onav_rnn.yaml
# configfile=offline_bc/config/onav_transformer.yaml
configfile=offline_bc/config/onav_imap_single_transformer.yaml

srun $python_bin offline_bc/train_models.py --exp-config $configfile

