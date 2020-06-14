def get_wp_matrix(ids, mts, wps, tokenizer, target_only=False):
    wp_matrix = []
    for id, mt_toks, word_probs in zip(ids, mts, wps):

        wp_matrix.append([0.0] * len(id))
        bert_toks = tokenizer.convert_ids_to_tokens(id)
        bert_i, mt_i, wp_i = 0, 0, 0

        #start after the sep token
        if target_only:
            bert_i = 1
        else:
            while bert_toks[bert_i] != tokenizer.sep_token:
                bert_i += 1
            bert_i += 1
            #hack for xlm-roberta
            if bert_toks[bert_i] == tokenizer.sep_token:
                bert_i += 1

        done = False
        next_bert_tok = bert_toks[bert_i]
        next_mt_tok = mt_toks[mt_i]
        debug = ''
        retry = 0
        while not done:
            next_bert_tok = next_bert_tok.replace("##", "").replace("</w>", "").replace("▁","")
            next_mt_tok = next_mt_tok.replace("@@", "").replace("▁","")

            #a hack to prevent this from getting into infinite loop
            if  "%s %s" % (next_bert_tok, next_mt_tok) == debug:
                if retry > 10:
                    mt_i += 1
                    wp_i += 1
                    if mt_i == len(mt_toks): 
                        break
                    else: 
                        next_mt_tok = mt_toks[mt_i]
                else:
                    retry += 1
            else:
                debug = "%s %s" % (next_bert_tok, next_mt_tok)
                retry = 0

            if next_bert_tok == tokenizer.unk_token:
                bert_i += 1
                mt_i += 1
                wp_i += 1
                if bert_i == len(bert_toks): done = True
                else: next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]

            elif next_mt_tok.startswith("&") and next_mt_tok.endswith(";"):
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                mt_i += 1
                wp_i += 1
                if bert_i == len(bert_toks): done = True
                else: next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]

            elif next_bert_tok == next_mt_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                mt_i += 1
                wp_i += 1
                if bert_i == len(bert_toks): done = True
                else: next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]

            elif next_bert_tok in next_mt_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                next_mt_tok = next_mt_tok.replace(next_bert_tok, '', 1)
                if bert_i == len(bert_toks): done = True
                else: next_bert_tok = bert_toks[bert_i]
                if not next_mt_tok:
                    mt_i += 1
                    wp_i += 1
                    if mt_i == len(mt_toks): done = True
                    else: next_mt_tok = mt_toks[mt_i]

            elif next_mt_tok in next_bert_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                mt_i += 1
                wp_i += 1
                next_bert_tok = next_bert_tok.replace(next_mt_tok, '', 1)
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]
    return wp_matrix
