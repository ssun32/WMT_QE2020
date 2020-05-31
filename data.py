import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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


def collate_fn(batches, tokenizer, use_word_probs=False, encode_separately=False):
    batch_text = []
    src_batch_text = []
    tgt_batch_text = []
    mts = []
    wps = []
    batch_z_scores = []
    batch_da_scores = []

    for batch in batches:
        src_batch_text.append(batch["source"])
        tgt_batch_text.append(batch["target"])
        batch_text.append((batch["source"], batch["target"]))
        mts.append(batch["mt"])
        wps.append(batch["wp"])

        batch_z_scores.append(batch["z_score"])
        batch_da_scores.append(batch["da_score"])

    if not encode_separately:
        tokenized = [tokenizer.batch_encode_plus(batch_text, max_length=220, pad_to_max_length=True, return_tensors = "pt")]
    else:
        tokenized = [tokenizer.batch_encode_plus(src_batch_text, max_length=64, pad_to_max_length=True, return_tensors = "pt"), tokenizer.batch_encode_plus(tgt_batch_text, max_length=64, pad_to_max_length=True, return_tensors = "pt")]

    if use_word_probs:
        wp_matrix = get_wp_matrix(tokenized[-1]["input_ids"].tolist(), mts, wps, tokenizer, target_only=encode_separately)
        wp_matrix = torch.tensor(wp_matrix)
    else:
        wp_matrix = None
    return tokenized, wp_matrix, batch_z_scores, batch_da_scores


class QEDataset(Dataset):
    def __init__(self, filepath, mt_filepath, wp_filepath):
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
                    mean_score = None if "mean" not in header else float(items[header["mean"]])/100
                    zmean_score = None if "z_mean" not in header else float(items[header["z_mean"]])

                    dataset.append({
                                   "source": items[header["original"]],
                                   "target": items[header["translation"]],
                                   "da_score": mean_score,
                                   "z_score": zmean_score})

            for i, (l, l2) in enumerate(zip(open(mtp), open(wpp))):
               dataset[i]["mt"] = l.strip().split()
               dataset[i]["wp"] = l2.strip().split()


            #random 50%
            #import random
            #random.seed(12345)
            #if len(dataset) > 1000:
            #    dataset = random.sample(dataset, int(len(dataset)/2))
            self.datasets += dataset

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        return self.datasets[index]

class QEDatasetRoundRobin(object):
    def __init__(self, filepath, mt_filepath, wp_filepath, batch_size, collate_fn, accum_grad=1):
        self.all_datasets = None
        self.plc_datasets = []
        self.accum_grad = accum_grad

        for fp, mtp, wpp in zip(filepath, mt_filepath, wp_filepath):
            dataloader = DataLoader(QEDataset(fp, mtp, wpp), 
                                    batch_size=batch_size, 
                                    collate_fn=collate_fn, 
                                    shuffle=True)

            if len(mtp) > 2:
                lcode = ("all", "all")
                self.all_datasets = iter(dataloader)
            else:
                lcode = mtp[-1].split(".")[-1]
                lcode = (lcode[:2], lcode[2:])
                self.plc_datasets.append((lcode, iter(dataloader)))


    def __iter__(self):
        all_iterator_ended = False
        plc_iterator_ended = 0
        cur_plc_dataset = 0
        while not all_iterator_ended and plc_iterator_ended < len(self.plc_datasets):
            all_tmp = []
            plc_tmp = []
            cur_lcode = self.plc_datasets[cur_plc_dataset][0]
            for i in range(self.accum_grad):
                try:
                    plc_tmp.append(next(self.plc_datasets[cur_plc_dataset][-1]))
                except StopIteration:
                    plc_iterator_ended += 1
                    break
                try:
                    all_tmp.append(next(self.all_datasets))
                except StopIteration:
                    all_iterator_ended = True

            for i, batch in enumerate(all_tmp):
                backprop = i==len(all_tmp)-1
                yield ("all", "all"), backprop, batch
            for i, batch in enumerate(plc_tmp):
                backprop = i==len(plc_tmp)-1
                yield cur_lcode, backprop, batch

            cur_plc_dataset += 1
            if cur_plc_dataset == len(self.plc_datasets):
                cur_plc_dataset = 0
