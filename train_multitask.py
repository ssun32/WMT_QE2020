import sys
import os
import torch
import json
import logging
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
parser.add_argument('--config_file', required=True)
parser.add_argument('--num_gpus', type=int, default=1)
args = parser.parse_args()
print(args)

with open(args.config_file) as fjson:
        config = json.load(fjson)

#get parameters from config file
model_name = config["model_name"]
model_dim = config["model_dim"]
learning_rate= config["learning_rate"]
epochs = config["epochs"]
batch_size = config["batch_size_per_gpu"] * args.num_gpus
use_word_probs = config["use_word_probs"]
use_secondary_loss = config["use_secondary_loss"]
accum_grad = config["accum_grad"]
eval_interval = config["eval_interval"]

#load model and optimizer
gpu=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = AutoModel.from_pretrained(model_name)

train_ids_list = []
train_file_list = []
train_mt_file_list = []
train_wp_file_list = []

dev_datasets, test_datasets = [], {}
dev_ids, test_ids = [], []
for train in config["train"]:
    train_ids_list.append(train["id"])
    train_file_list.append(train["tsv_file"])
    train_mt_file_list.append(train["mt_file"])
    train_wp_file_list.append(train["wp_file"])

for dev in config["dev"]:
    dev_ids.append(dev["id"])
    dev_file = dev["tsv_file"]
    dev_mt_file = dev["mt_file"]
    dev_wp_file = dev["wp_file"]
    dev_datasets.append((dev["id"], QEDataset(dev_file, dev_mt_file, dev_wp_file)))

for test in config["test"]:
    test_ids.append(test["id"])
    test_file = test["tsv_file"]
    test_mt_file = test["mt_file"]
    test_wp_file = test["wp_file"]
    test_datasets[test["id"]] = (test["id"], QEDataset(test_file, test_mt_file, test_wp_file))

#make a roundrobin dataset
collate_fn = partial(collate_fn, tokenizer=tokenizer, use_word_probs=use_word_probs)

#create model and optimizer
model = QE(transformer, model_dim, train_ids_list, use_word_probs=use_word_probs)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(gpu)

loss_fn = torch.nn.MSELoss()

log_file = os.path.join(config["output_dir"], "log")
logging.basicConfig(format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
			    handlers=[
				logging.FileHandler(log_file),
				logging.StreamHandler()
			    ],
                            level=logging.DEBUG)

def eval(dataset, id, get_metrics=False):
    with torch.no_grad():
        model.eval()

        predicted_scores, actual_scores = [], []
        for batch, wps, z_scores, _ in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, use_word_probs = use_word_probs), shuffle=False)):
            batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
            wps = wps.to(gpu) if wps is not None else wps

            #force nan to be 0, this deals with bad inputs from si-en dataset
            z_score_outputs, _ = model(batch, wps, id)
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

        model.train()
        return predicted_scores, pearson, mse

global_steps = 0
best_eval = 0
early_stop = 0
best_eval_per_lang = {id: 0 for id in dev_ids}
for epoch in range(epochs):
    print("Epoch ", epoch)
    total_loss = 0
    total_batches = 0
    train_dataset = QEDatasetRoundRobin(train_ids_list, train_file_list, train_mt_file_list, train_wp_file_list, batch_size, collate_fn, accum_grad=accum_grad)

    for cur_id, backprop, (batch, wps, z_scores, da_scores) in tqdm(train_dataset):
        batch = [{k: v.to(gpu) for k, v in b.items()} for b in batch]
        wps = wps.to(gpu) if wps is not None else wps
        z_scores = torch.tensor(z_scores).to(gpu)
        z_score_outputs, joint_output = model(batch, wps, cur_id)

        #drop batch with nan
        if torch.isnan(z_score_outputs).any():
            if backprop:
                model.zero_grad()
            del batch, wps, z_scores, z_score_outputs
            continue

        loss = loss_fn(z_score_outputs.squeeze(), z_scores)
        #if cur_id != "all":
        #    loss += loss_fn(joint_output.squeeze(), z_scores)
        cur_batch_size = z_score_outputs.size(0)

        total_loss += loss.item() * cur_batch_size
        total_batches += cur_batch_size

        #back prob
        #if global_steps % accum_grad == 0:
        if backprop:
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_steps += 1

        del batch, wps, z_scores, z_score_outputs, loss

        if backprop and global_steps % eval_interval == 0:
            logging.info("\nCalculating results on dev sets(s)... (lang specific MLP)")
            #eval on lang spec MLP layer
            for (id, dev_dataset) in dev_datasets:
                if id not in train_ids_list:
                    continue
                predicted_scores, pearson, mse =  eval(dev_dataset, id, get_metrics=True)
                if pearson > best_eval_per_lang[id]:
                    best_eval_per_lang[id] = pearson
                    best_dev_file = os.path.join(config["output_dir"], "%s.lang_spec_mlp.dev.best.scores"%id)
                    with open(best_dev_file, "w") as fout:
                        for score in predicted_scores:
                            print(score, file=fout)

                    if id in test_datasets:
                        _, test_dataset = test_datasets[id]
                        predicted_scores, _, _ = eval(test_dataset, id)
                        best_test_file = os.path.join(config["output_dir"], "%s.lang_spec_mlp.test.best.scores"%id)
                        with open(best_test_file, "w") as fout:
                            for score in predicted_scores:
                                print(score, file=fout)
                    logging.info("\nid:%s cur r: %.4f best r: %.4f" % (id, pearson, best_eval_per_lang[id]))

            #eval on lang agnost MLP 
            dev_results = []
            total_pearson, total = 0, 0
            logging.info("\nCalculating results on dev set(s)... (lang agnostic mlp)")
            for id, dev_dataset in dev_datasets:
                predicted_scores, pearson, mse =  eval(dev_dataset, "all", get_metrics=True)
                dev_results.append((id, predicted_scores, pearson, mse))
                total_pearson += pearson
                total += 1

            avg_pearson = total_pearson/total
            if avg_pearson > best_eval:
                best_eval = avg_pearson
                print()
                for id, predicted_scores, _, _ in dev_results:
                    best_dev_file = os.path.join(config["output_dir"], "%s.lang_agnost_mlp.dev.best.scores"%id)
                    logging.info("Saving best dev results to: %s" % best_dev_file)
                    with open(best_dev_file, "w") as fout:
                        for score in predicted_scores:
                            print(score, file=fout)

                test_results = []
                print("\nCalculating results on test set(s)...")
                for id in test_datasets:
                    _, test_dataset = test_datasets[id]
                    predicted_scores, _, _ =  eval(test_dataset, "all")
                    test_results.append((id, predicted_scores))

                for id, predicted_scores in test_results:
                    best_test_file = os.path.join(config["output_dir"], "%s.lang_agnost_mlp.test.best.scores"%id)
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
            for id, _, pearson, mse in dev_results:
                log +="%s Dev loss: %.4f r:%.4f\n" % (id, mse, pearson)
            log +="Current avg r:%.4f Best avg r: %.4f" % (avg_pearson, best_eval)
            logging.info(log)
        if early_stop > 200:
            break
    if early_stop > 200:
        break
