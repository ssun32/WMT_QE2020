import sys
import torch
import numpy as np
from data import QEDataset, QEDatasetRoundRobin, collate_fn
from model_multitask import QE
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from functools import partial
from scipy.stats import pearsonr
from glob import glob
from tqdm import tqdm
import argparse

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default="bert")
parser.add_argument('--output_prefix', required=True)
parser.add_argument('--use_word_probs', nargs="?", const=True, default=False)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()
print(args)

#model specific configuration
warmup_steps = 5000
if args.model.lower() == "xlm":
    model_name = "xlm-mlm-100-1280"
    model_dim = 1280
    learning_rate = 1e-6
    batch_size = 6 * args.num_gpus
    eval_interval = 100

elif args.model.lower() == "xlm_roberta":
    model_name = "xlm-roberta-base"
    model_dim=  768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100

elif args.model.lower() == "xlm_roberta_large":
    model_name = "xlm-roberta-large"
    model_dim=  1024
    learning_rate = 1e-6
    batch_size = 8 * args.num_gpus
    eval_interval = 500
else:
    model_name = "bert-base-multilingual-cased"
    model_dim = 768
    learning_rate = 1e-6
    batch_size = 16 * args.num_gpus
    eval_interval = 100
    if args.use_word_probs:
        batch_size = 12

#load model and optimizer
gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

filedir = "data/*"
train_file_list = [glob("%s/train*.tsv" % filedir)]
train_mt_file_list = [glob("%s/word-probas/mt.train*" % filedir)]
train_wp_file_list = [glob("%s/word-probas/word_probas.train*" % filedir)]

lcodes = [("en","de"), ("en","zh"), ("ro","en"), ("et","en"), ("si","en"), ("ne","en"), ("ru", "en")]

dev_datasets, test_datasets = [], []
for src_lcode, tgt_lcode in lcodes:
    filedir = "data/%s-%s"%(src_lcode, tgt_lcode)
    train_file_list.append(glob("%s/train*.tsv" % filedir))
    train_mt_file_list.append(glob("%s/word-probas/mt.train*" % filedir))
    train_wp_file_list.append(glob("%s/word-probas/word_probas.train*" % filedir))

    dev_file = glob("%s/dev*.tsv" % filedir)
    dev_mt_file = glob("%s/word-probas/mt.dev*" % filedir)
    dev_wp_file = glob("%s/word-probas/word_probas.dev*" % filedir)
    dev_datasets.append(((src_lcode, tgt_lcode), QEDataset(dev_file, dev_mt_file, dev_wp_file)))

    test_file = glob("%s/test20*.tsv" % filedir)
    test_mt_file = glob("%s/word-probas/mt.test20*" % filedir)
    test_wp_file = glob("%s/word-probas/word_probas.test20*" % filedir)
    test_datasets.append(((src_lcode, tgt_lcode), QEDataset(test_file, test_mt_file, test_wp_file)))

#make a roundrobin dataset
collate_fn = partial(collate_fn, tokenizer=tokenizer, use_word_probs=args.use_word_probs)

#create model and optimizer
model = QE(transformer, model_dim, lcodes, use_word_probs=args.use_word_probs)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(gpu)
#move mlp layers to cpu
model.mlp_layers = model.mlp_layers.to("cpu")

loss_fn = torch.nn.MSELoss()

log_file = args.output_prefix + ".log"
flog = open(log_file, "w")

def eval(dataset, lcode, get_metrics=False):
    with torch.no_grad():
        model.eval()
        module_lcode = "_".join(lcode)
        model.mlp_layers[module_lcode] = model.mlp_layers[module_lcode].to(gpu)

        predicted_scores, actual_scores = [], []
        for batch, wps, z_scores, _ in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, use_word_probs = args.use_word_probs), shuffle=False)):
            batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
            wps = wps.to(gpu) if wps is not None else wps

            #force nan to be 0, this deals with bad inputs from si-en dataset
            z_score_outputs = model(batch, wps, lcode)
            z_score_outputs[torch.isnan(z_score_outputs)] = 0 
            predicted_scores += z_score_outputs.flatten().tolist()
            actual_scores += z_scores
        if get_metrics:
            predicted_scores = np.array(predicted_scores)
            actual_scores = np.array(actual_scores)
            pearson = pearsonr(predicted_scores, actual_scores)[0]
            mse = np.square(np.subtract(predicted_scores, actual_scores)).mean()
        else:
            pearson, mse = None, None
            
            del batch, wps, z_score_outputs

        model.mlp_layers[module_lcode] = model.mlp_layers[module_lcode].to("cpu")
        model.train()
        return predicted_scores, pearson, mse

global_steps = 0
best_eval = 0
best_eval_per_lang = {lcode: 0 for lcode in lcodes}
early_stop = 0
for epoch in range(50):
    print("Epoch ", epoch)
    total_loss = 0
    total_batches = 0
    train_dataset = QEDatasetRoundRobin(train_file_list, train_mt_file_list, train_wp_file_list, batch_size, collate_fn)

    for cur_lcode, (batch, wps, z_scores, da_scores) in tqdm(train_dataset):
        global_steps += 1
        module_lcode = "_".join(cur_lcode)
        model.mlp_layers[module_lcode] = model.mlp_layers[module_lcode].to(gpu)
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps
        z_scores = torch.tensor(z_scores).to(gpu)
        z_score_outputs = model(batch, wps, cur_lcode)

        #drop batch with nan
        if torch.isnan(z_score_outputs).any():
            del batch, wps, z_scores, z_score_outputs
            model.mlp_layers[module_lcode] = model.mlp_layers[module_lcode].to("cpu")
            continue

        loss = loss_fn(z_score_outputs.squeeze(), z_scores)
        cur_batch_size = z_score_outputs.size(0)

        total_loss += loss.item() * cur_batch_size
        total_batches += cur_batch_size

        #back prob
        loss.backward()
        del batch, wps, z_scores, z_score_outputs, loss
        model.mlp_layers["_".join(module_lcode)] = model.mlp_layers[module_lcode].to("cpu")

        optimizer.step()
        optimizer.zero_grad()

        if global_steps % eval_interval == 0 and global_steps > warmup_steps:
            #eval on per_lang MLP layer
            for (lcode, dev_dataset), (_, test_dataset) in zip(dev_datasets, test_datasets):
                predicted_scores, pearson, mse =  eval(dev_dataset, lcode, get_metrics=True)
                if pearson > best_eval_per_lang[lcode]:
                    best_eval_per_lang[lcode] = pearson
                    best_dev_file = args.output_prefix + ".%s%s.perlangmlp.dev.best.scores"%lcode
                    with open(best_dev_file, "w") as fout:
                        for score in predicted_scores:
                            print(score, file=fout)

                    predicted_scores, _, _ = eval(test_dataset, lcode)
                    best_test_file = args.output_prefix + ".%s%s.perlangmlp.test.best.scores"%lcode
                    with open(best_test_file, "w") as fout:
                        for score in predicted_scores:
                            print(score, file=fout)
                print("\n(%s, %s): cur r: %.4f best r: %.4f" % (lcode[0], lcode[1], pearson, best_eval_per_lang[lcode]))

            #eval on all_lang MLP 
            dev_results = []
            total_pearson, total = 0, 0
            print("\nCalculating results on dev set(s)...")
            for lcode, dev_dataset in dev_datasets:
                predicted_scores, pearson, mse =  eval(dev_dataset, ("all", "all"), get_metrics=True)
                dev_results.append((lcode, predicted_scores, pearson, mse))
                total_pearson += pearson
                total += 1

            avg_pearson = total_pearson/total
            if avg_pearson > best_eval:
                best_eval = avg_pearson
                print()
                for lcode, predicted_scores, _, _ in dev_results:
                    best_dev_file = args.output_prefix + ".%s%s.alllangmlp.dev.best.scores"%lcode
                    print("Saving best dev results to: %s" % best_dev_file)
                    with open(best_dev_file, "w") as fout:
                        for score in predicted_scores:
                            print(score, file=fout)

                test_results = []
                print("\nCalculating results on test set(s)...")
                for lcode, test_dataset in test_datasets:
                    predicted_scores, _, _ =  eval(test_dataset, ("all", "all"))
                    test_results.append((lcode, predicted_scores))

                for lcode, predicted_scores in test_results:
                    best_test_file = args.output_prefix + ".%s%s.alllangmlp.test.best.scores"%lcode
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
            for lcode, _, pearson, mse in dev_results:
                log +="%s-%s Dev loss: %.4f r:%.4f\n" % (lcode[0], lcode[1], mse, pearson)
            log +="Current avg r:%.4f Best avg r: %.4f" % (avg_pearson, best_eval)
            print(log)
            print(log, file=flog)
        if early_stop > 25:
            break
    if early_stop > 25:
        break
