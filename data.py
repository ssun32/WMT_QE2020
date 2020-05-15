import torch
from torch.utils.data import Dataset

def get_wp_matrix(ids, mts, wps, tokenizer, target_only=False):
    wp_matrix = []
    for id, mt_toks, word_probs in zip(ids, mts, wps):

        wp_matrix.append([0] * len(id))
        bert_toks = tokenizer.convert_ids_to_tokens(id)
        bert_i, mt_i, wp_i = 0, 0, 0

        #start after the sep token
        if target_only:
            bert_i = 1
        else:
            while bert_toks[bert_i] != tokenizer.sep_token:
                bert_i += 1
            bert_i += 1

        done = False
        next_bert_tok = bert_toks[bert_i]
        next_mt_tok = mt_toks[mt_i]
        debug = ''
        retry = 0
        while not done:
            next_bert_tok = next_bert_tok.replace("##", "").replace("</w>", "")
            next_mt_tok = next_mt_tok.replace("@@", "").replace("▁","")
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
                next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): 
                    done = True
                else: 
                    next_mt_tok = mt_toks[mt_i]

            elif next_mt_tok.startswith("&") and next_mt_tok.endswith(";"):
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                mt_i += 1
                wp_i += 1
                next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]
            elif next_bert_tok == next_mt_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                mt_i += 1
                wp_i += 1
                next_bert_tok = bert_toks[bert_i]
                if mt_i == len(mt_toks): done = True
                else: next_mt_tok = mt_toks[mt_i]
            elif next_bert_tok in next_mt_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                bert_i += 1
                next_mt_tok = next_mt_tok.replace(next_bert_tok, '', 1)
                next_bert_tok = bert_toks[bert_i]
                if not next_mt_tok:
                    mt_i += 1
                    wp_i += 1
                    if mt_i == len(mt_toks): 
                        done = True
                    else: 
                        next_mt_tok = mt_toks[mt_i]
            elif next_mt_tok in next_bert_tok:
                wp_matrix[-1][bert_i] = float(word_probs[wp_i])
                mt_i += 1
                wp_i += 1
                next_bert_tok = next_bert_tok.replace(next_mt_tok, '', 1)
                if mt_i == len(mt_toks): 
                    done = True
                else: 
                    next_mt_tok = mt_toks[mt_i]
    return wp_matrix


def collate_fn(batches, tokenizer, use_word_probs=False, encode_separately=False):
    batch_text = []
    src_batch_text = []
    tgt_batch_text = []
    mts = []
    wps = []
    batch_scores = []
    for batch in batches:
        src_batch_text.append(batch["source"])
        tgt_batch_text.append(batch["target"])
        batch_text.append((batch["source"], batch["target"]))
        mts.append(batch["mt"])
        wps.append(batch["wp"])
        batch_scores.append(batch["score"])

    if not encode_separately:
        tokenized = [tokenizer.batch_encode_plus(batch_text, max_length=256, pad_to_max_length=True, return_tensors = "pt")]
    else:
        tokenized = [tokenizer.batch_encode_plus(src_batch_text, max_length=128, pad_to_max_length=True, return_tensors = "pt"), tokenizer.batch_encode_plus(tgt_batch_text, max_length=128, pad_to_max_length=True, return_tensors = "pt")]

    if use_word_probs:
        wp_matrix = get_wp_matrix(tokenized[-1]["input_ids"].tolist(), mts, wps, tokenizer, target_only=encode_separately)
        wp_matrix = torch.tensor(wp_matrix)
    else:
        wp_matrix = None
    return tokenized, wp_matrix, batch_scores


class QEDataset(Dataset):
    def __init__(self, filepath, mt_filepath, wp_filepath, score_field=None):
        if type(filepath) == type("str"):
            filepath = [filepath]
            mt_filepath = [mt_filepath]
            wp_filepath = [wp_filepath]

        self.datasets = []
        for fp, mtp, wpp in zip(filepath, mt_filepath, wp_filepath):
            dataset = []

            for i, l in enumerate(open(fp)):
                if i == 0:
                    header = {h:j for j,h in enumerate(l.strip().split("\t"))}
                else:
                    items = l.strip().split("\t")
                    score = None if score_field is None else float(items[header[score_field]])
                    dataset.append({"source": items[header["original"]],
                                   "target": items[header["translation"]],
                                   "score": score})

            for i, (l, l2) in enumerate(zip(open(mtp), open(wpp))):
               dataset[i]["mt"] = l.strip().split()
               dataset[i]["wp"] = l2.strip().split()
            self.datasets += dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]
