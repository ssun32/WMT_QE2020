#python train --src en --tgtzh --output_prefix enzh --model bert

for model in bert xlm xlm_roberta;
    do

    #without word_probas
    python train.py --src en --tgt de --output_prefix outputs/ende_${model} --model $model
    python train.py --src en --tgt zh --output_prefix outputs/enzh_${model} --model $model
    python train.py --src si --tgt en --output_prefix outputs/sien_${model} --model $model
    python train.py --src ne --tgt en --output_prefix outputs/neen_${model} --model $model
    python train.py --src et --tgt en --output_prefix outputs/eten_${model} --model $model
    python train.py --src ro --tgt en --output_prefix outputs/roen_${model} --model $model

    #with word_probas
    python train.py --src en --tgt de --output_prefix outputs/ende_${model}_wp --model $model --use_word_probs
    python train.py --src en --tgt zh --output_prefix outputs/enzh_${model}_wp --model $model --use_word_probs
    python train.py --src si --tgt en --output_prefix outputs/sien_${model}_wp --model $model --use_word_probs
    python train.py --src ne --tgt en --output_prefix outputs/neen_${model}_wp --model $model --use_word_probs
    python train.py --src et --tgt en --output_prefix outputs/eten_${model}_wp --model $model --use_word_probs
    python train.py --src ro --tgt en --output_prefix outputs/roen_${model}_wp --model $model --use_word_probs
    done
