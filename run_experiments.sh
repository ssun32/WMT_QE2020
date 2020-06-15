num_gpus=1

for f in experiments/*/*/*/config.json; do
    if [[ $f == *"mtl"* ]]; then
        python train_multitask.py --config_file $f --num_gpus=$num_gpus
    else
        python train.py --config_file $f --num_gpus=$num_gpus
    fi
