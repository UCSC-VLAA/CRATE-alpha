#!/bin/bash

export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME


TFDS_DATA_DIR=[your ImageNet-1k dataset location]
DATACOMP1B_PATH=[your datacomp1b dataset location]
WORK_DIR=[your work dir to save ckpt]
WANDB_log=[your wandb login key] # only if you set wandb.log_wandb=True then you can revise the project name and experiment name

SAVE_NAME=[The name of the task dir]
MASK_INIT=[The path of the checkpoint for the text encoder]
config=configs/crate/clipa/84_32_pre_training.py


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd ~/CRATE-alpha/ &&  . bv_venv/bin/activate && \
wandb login  $WANDB_log && cd ~/CRATE-alpha/ && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} python3 -m main_clip \
--config=$config:img=L/14 \
--workdir=$WORK_DIR \
--config.masked_init=$MASK_INIT \
--config.input.data.data_dir=$DATACOMP1B_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR \
--config.wandb.experiment=crate_H14_datacomp${SAVE_NAME} "

