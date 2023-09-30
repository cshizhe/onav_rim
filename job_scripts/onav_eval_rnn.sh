#!/bin/bash
#SBATCH --job-name=onav_eval
#SBATCH -p gpu_p13               # Name of the partition
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=2      # Number of GPUs on a node
#SBATCH --gres gpu:2             # The same
#SBATCH -c 10                    # Number of workers per GPU
#SBATCH --hint=nomultithread     # Logical cores
#SBATCH --time 20:00:00          # Total run time limit (HH:MM:SS)
#SBATCH --output=slurm_logs/%j.out # Output file name
#SBATCH --error=slurm_logs/%j.out  # Error file name


cd /home/shichen/codes/onav_rim

export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x

outdir=$1
ckpt_step=$2
result_dir=$outdir/step_${ckpt_step}_heuristic_nocld

configpath=habitat_baselines/config/objectnav/eval_rnn.yaml
val_dataset_path=data/datasets/objectnav/mp3d/v1
checkpoint=$outdir/ckpts/model_step_${ckpt_step}.pt
# evalsplit=minival # minival, val
evalsplit=val

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

python_bin=$HOME/miniconda3/envs/onav/bin/python

srun $python_bin -u habitat_baselines/run.py \
--exp-config  $configpath \
--run-type eval \
TASK_CONFIG.DATASET.DATA_PATH "$val_dataset_path/{split}/{split}.json.gz" \
TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']" \
EVAL_CKPT_PATH_DIR $checkpoint \
CHECKPOINT_FOLDER $outdir/ckpts OUTPUT_LOG_DIR $outdir/logs \
LOG_FILE $result_dir/valid.log \
TENSORBOARD_DIR ${result_dir}/tb VIDEO_DIR ${result_dir}/video_dir \
RESULTS_DIR ${result_dir}/results/sem_seg_pred/{split}/{type} \
EVAL_RESULTS_DIR ${result_dir}/results \
EVAL.USE_CKPT_CONFIG False \
EVAL.SPLIT ${evalsplit} \
NUM_PROCESSES 1 \
EVAL_CKPT_FROM_OFFLINEBC True \
MODEL.enc_collide_steps False 
