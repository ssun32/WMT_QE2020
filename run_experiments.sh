#!/bin/bash
set -e
num_gpus=1
job_name="H1.05"
for f in experiments_xlmr/${job_name}/*/*/config.json; do
    echo $f
    suffix="config.json"
    scores=${f%"$suffix"}

    #if ls ${scores}*.test.best.scores 1> /dev/null 2>&1; then
    #    echo "files do exist"
    #else
        if [[ $f == *"mtl"* ]]; then
        srun --job-name $f --partition=priority --comment="AACl 06/26" --error ${scores}.run.stderr --output ${scores}.run.stdout --time=2880 --nodes=1 --gpus-per-node=$num_gpus --cpus-per-task 8 --constraint="volta" python train_multitask.py --config_file $f --num_gpus=$num_gpus &
        else
        srun --job-name $f --partition=priority --comment="AACl 06/26" --error ${scores}.run.stderr --output ${scores}.run.stdout --time=2880 --nodes=1 --gpus-per-node=$num_gpus --cpus-per-task 8 --constraint="volta" python train.py --config_file $f --num_gpus=$num_gpus &
        fi
    #fi
done
