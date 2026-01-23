#!/bin/bash

#SBATCH -p gpus
#SBATCH --gres gpu:1
#SBATCH --nodelist lory
#SBATCH --output=run_logs/train_slurm.%N.%j.log

source /vol/biomedic3/awk24/miniconda3/bin/activate
conda activate flow-env

export RDMAV_FORK_SAFE=1
export OPENAI_LOG_FORMAT="stdout,log,csv,tensorboard"
export OPENAI_LOGDIR="/vol/biomedic3/awk24/code/classifier/outputs_all_classes"

# inception_v3.tv_in1k
# efficientnet_b0
# python train.py  --learning_rate 0.001  --num_epoch 100 --model_name "inception_v3.tv_in1k" --augment True --augment_with_ai True

# python train.py  --learning_rate 0.001  --num_epoch 40 --model_name "inception_v3.tv_in1k" --augment True --augment_with_ai True --skip_early_glaucoma True --num_per_class 15
# python train.py  --learning_rate 0.001  --num_epoch 70 --model_name "inception_v3.tv_in1k" --augment True --augment_with_ai False --skip_early_glaucoma False --num_per_class 1

# python test_models.py
# python test_classifier_2_class.py --num 200
# python classify_generated_2_class.py
python test_classifier.py --num 1