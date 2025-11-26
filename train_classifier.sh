#!/bin/bash

#SBATCH -p gpus48
#SBATCH --gres gpu:1
#SBATCH --nodelist luna
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
conda activate diffusion

export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/classifier/outputs"

# inception_v3.tv_in1k
# efficientnet_b0
python train.py  --learning_rate 0.001  --num_epoch 50 --model_name "inception_v3.tv_in1k"
# python test_models.py
