export PROJECT_ID=[your project id]
export ZONE=[your TPU location]
export TPU_NAME=[your TPU VM name]


TFDS_DATA_DIR=[your ImageNet-1k dataset location]
LAION_PATH=[your laion-400m dataset location]
WORK_DIR=[your work dir to save ckpt]
WANDB_log=[your wandb login key] # only if you set wandb.log_wandb=True then you can revise the project name and experiment name


SAVE_NAME=[The name of the task dir]
MODEL=B/16
config=configs/crate/classification/pre_training_in21k.py


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd ~/CRATE-alpha/ &&  . bv_venv/bin/activate  &&   wandb login $WANDB_log && \
cd ~//CRATE-alpha/ && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} python3 -m main \
--config=$config:variant=${MODEL} \
--workdir=$WORK_DIR \
--config.model.drop_path=0. \
--config.wandb.experiment=crate_${MODEL}_${SAVE_NAME} "