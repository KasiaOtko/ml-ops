#!/bin/sh
dvc pull
python -u src/models/train_model_no_wandb.py $1