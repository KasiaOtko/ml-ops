#!/bin/sh
dvc pull
wandb login $1
python -u src/models/train_model_no_wandb.py $2