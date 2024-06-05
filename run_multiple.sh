#!/bin/bash

# Define an array of commands
commands=(
    "python align_fine_tuning.py --batch_size 20 --accumulation_steps 8 --save --save_every 1000 --verbose --loss_func contrastive --split_name fold_0_t --stop_patience 0 --maximize --epochs 10000 --lr 0.00000001"
    # "python align_fine_tuning.py --batch_size 20 --accumulation_steps 8 --save --save_every 50 --verbose --loss_func contrastive --split_name fold_0_t --stop_patience 20 --maximize --epochs 1000 --lr 0.00000005"
)

# Loop over commands and execute each one
for cmd in "${commands[@]}"
do
    echo "Executing: $cmd"
    $cmd
    echo "Finished: $cmd"
    echo
done