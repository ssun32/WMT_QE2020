import sys
import torch
import numpy as np
from data import QEDataset, collate_fn
from model import QE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--src', default="en")
parser.add_argument('--tgt', default="de")
parser.add_argument('--model', default="bert")
parser.add_argument('--output_prefix', required=True)
parser.add_argument('--use_word_probs', nargs="?", const=True, default=False)
parser.add_argument('--encode_separately', nargs="?", const=True, default=False)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()
print(args)

src_lcode = args.src
tgt_lcode = args.tgt

#model specific configuration
if args.model.lower() == "xlm":
    model_name = "xlm-mlm-100-1280"
    model_dim = 1280
    learning_rate = 1e-6
    batch_size = 6 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
elif args.model.lower() == "xlm_roberta":
    model_name = "xlm-roberta-base"
    model_dim=  768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
elif args.model.lower() == "xlm_roberta_large":
    model_name = "xlm-roberta-large"
    model_dim=  1024
    learning_rate = 1e-6
    batch_size = 6 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
else:
    model_name = "bert-base-multilingual-cased"
    model_dim = 768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100
    accum_grad = 1
    if args.use_word_probs:
        batch_size = 12

#load model and optimizer
gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

model = QE(transformer, model_dim, use_word_probs = args.use_word_probs, encode_separately=args.encode_separately)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(gpu)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

filedir = "data/%s-%s"%(src_lcode, tgt_lcode) if src_lcode != "all" else "data/*"
train_file = glob("%s/train*.tsv" % filedir)
train_mt_file = glob("%s/word-probas/mt.train*" % filedir)
train_wp_file = glob("%s/word-probas/word_probas.train*" % filedir)
train_dataset = QEDataset(train_file, train_mt_file, train_wp_file, score_field="z_mean")


if src_lcode == "all":
    lcodes = [("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en")]
else:
    lcodes = [(src_lcode, tgt_lcode)]

dev_datasets, test_datasets = [], []
for src_lcode, tgt_lcode in lcodes:
    filedir = "data/%s-%s"%(src_lcode, tgt_lcode)
    dev_file = glob("%s/dev*.tsv" % filedir)
    dev_mt_file = glob("%s/word-probas/mt.dev*" % filedir)
    dev_wp_file = glob("%s/word-probas/word_probas.dev*" % filedir)
    dev_datasets.append(((src_lcode, tgt_lcode), QEDataset(dev_file, dev_mt_file, dev_wp_file, score_field="z_mean")))

    test_file = glob("%s/test20*.tsv" % filedir)
    test_mt_file = glob("%s/word-probas/mt.test20*" % filedir)
    test_wp_file = glob("%s/word-probas/word_probas.test20*" % filedir)
    test_datasets.append(((src_lcode, tgt_lcode), QEDataset(test_file, test_mt_file, test_wp_file, score_field=None)))

log_file = args.output_prefix + ".log"
flog = open(log_file, "w")

def eval(dataset, get_metrics=False):
    model.eval()
    predicted_scores, actual_scores = [], []
    for batch, wps, labels in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, use_word_probs = args.use_word_probs, encode_separately=args.encode_separately), shuffle=False)):
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps

        #force nan to be 0, this deals with bad inputs from si-en dataset
        outputs = model(batch, wps)
        outputs[torch.isnan(outputs)] = 0 
        predicted_scores += outputs.flatten().tolist()

        actual_scores += labels
    if get_metrics:
        predicted_scores = np.array(predicted_scores)
        actual_scores = np.array(actual_scores)
        pearson = pearsonr(predicted_scores, actual_scores)[0]
        mse = np.square(np.subtract(predicted_scores, actual_scores)).mean()
    else:
        pearson, mse = None, None
    model.train()
    return predicted_scores, pearson, mse

global_steps = 0
best_eval = 0
early_stop = 0
accum_counter = 0
for epoch in range(50):
    print("Epoch ", epoch)
    total_loss = 0
    total_batches = 0
    for batch, wps, labels in tqdm(DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, use_word_probs=args.use_word_probs, encode_separately=args.encode_separately), shuffle=True)):
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps
        labels = torch.tensor(labels).to(gpu)
        outputs = model(batch, wps).squeeze()

        #drop batch with nan
        if torch.isnan(outputs).any():
            continue

        loss = loss_fn(outputs.squeeze(), labels)
        cur_batch_size = labels.size(0)
        total_loss += loss.item() * cur_batch_size
        total_batches += cur_batch_size
        loss.backward()
        del loss
        accum_counter += 1

        if accum_counter % accum_grad == 0:
            optimizer.step()
            optimizer.zero_grad()
            accum_counter = 0

        global_steps += 1

        with torch.no_grad():
            if global_steps % eval_interval == 0:
                dev_results = []
                total_pearson, total = 0, 0
                print("\nCalculating results on dev set(s)...")
                for lcodes, dev_dataset in dev_datasets:
                    predicted_scores, pearson, mse =  eval(dev_dataset, get_metrics=True)
                    dev_results.append((lcodes, predicted_scores, pearson, mse))
                    total_pearson += pearson
                    total += 1

                avg_pearson = total_pearson/total
                if avg_pearson > best_eval:
                    best_eval = avg_pearson
                    print()
                    for lcodes, predicted_scores, _, _ in dev_results:
                        best_dev_file = args.output_prefix + ".%s%s.dev.best.scores"%lcodes
                        print("Saving best dev results to: %s" % best_dev_file)
                        with open(best_dev_file, "w") as fout:
                            for score in predicted_scores:
                                print(score, file=fout)

                    test_results = []
                    print("\nCalculating results on test set(s)...")
                    for lcodes, test_dataset in test_datasets:
                        predicted_scores, _, _ =  eval(test_dataset)
                        test_results.append((lcodes, predicted_scores))

                    for lcodes, predicted_scores in test_results:
                        best_test_file = args.output_prefix + ".%s%s.test.best.scores"%lcodes
                        print("Saving best test results to: %s" % best_test_file)
                        with open(best_test_file, "w") as fout:
                            for score in predicted_scores:
                                print(score, file=fout)
                    early_stop = 0
                else:
                    early_stop += 1

                log = "Epoch %s Global steps: %s Train loss: %.4f\n" %(epoch, global_steps, total_loss/total_batches)
                #reset total loss 
                total_loss, total_batches = 0, 0
                for lcodes, _, pearson, mse in dev_results:
                    log +="%s-%s Dev loss: %.4f r:%.4f\n" % (lcodes[0], lcodes[1], mse, pearson)
                log +="Current avg r:%.4f Best avg r: %.4f" % (avg_pearson, best_eval)
                print(log)
                print(log, file=flog)
        if early_stop > 25:
            break
    if early_stop > 25:
        break
