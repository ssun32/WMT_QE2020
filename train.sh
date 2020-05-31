#python train --src en --tgtzh --output_prefix enzh --model bert

num_gpus=1
for model in xlm_roberta_large;
    do

    #with word_probas
    python train.py --src en --tgt de --output_prefix outputs/ende_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus 
    python train.py --src en --tgt zh --output_prefix outputs/enzh_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus
    python train.py --src si --tgt en --output_prefix outputs/sien_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus
    python train.py --src ne --tgt en --output_prefix outputs/neen_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus
    python train.py --src et --tgt en --output_prefix outputs/eten_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus
    python train.py --src ro --tgt en --output_prefix outputs/roen_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus
    python train.py --src ru --tgt en --output_prefix outputs/ruen_${model}_wp --model $model --use_word_probs --num_gpus $num_gpus

    #without word_probas
    #python train.py --src en --tgt de --output_prefix outputs/ende_${model}_half --model $model --num_gpus $num_gpus
    #python train.py --src en --tgt zh --output_prefix outputs/enzh_${model}_half --model $model --num_gpus $num_gpus
    #python train.py --src si --tgt en --output_prefix outputs/sien_${model}_half --model $model --num_gpus $num_gpus 
    #python train.py --src ne --tgt en --output_prefix outputs/neen_${model}_half --model $model --num_gpus $num_gpus
    #python train.py --src et --tgt en --output_prefix outputs/eten_${model}_half --model $model --num_gpus $num_gpus
    #python train.py --src ro --tgt en --output_prefix outputs/roen_${model}_half --model $model --num_gpus $num_gpus
    #python train.py --src ru --tgt en --output_prefix outputs/ruen_${model}_half --model $model --num_gpus $num_gpus


    done
