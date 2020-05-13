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

gpu="cuda:0" if torch.cuda.is_available() else "cpu"

#load model and optimizer
tokenizer = AutoTokenizer.from_pretrained("xlm-mlm-100-1280")
transformer = AutoModel.from_pretrained("xlm-mlm-100-1280")

model = QE(transformer, 1280).to(gpu)
model.train()
learning_rate = 1e-5
batch_size = 8
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

filedir = "data/en-de"
train_file = glob("%s/train*.tsv" % filedir)[0]
dev_file = glob("%s/dev*.tsv" % filedir)[0]
train_mt_file = glob("%s/word-probas/mt.train*" % filedir)[0]
dev_mt_file = glob("%s/word-probas/mt.dev*" % filedir)[0]
train_wp_file = glob("%s/word-probas/word_probas.train*" % filedir)[0]
dev_wp_file = glob("%s/word-probas/word_probas.dev*" % filedir)[0]

train_dataset = QEDataset(train_file, train_mt_file, train_wp_file, score_field="mean")
dev_dataset = QEDataset(dev_file, dev_mt_file, dev_wp_file, score_field="mean")

def eval(dataset):
    model.eval()
    predicted_scores, actual_scores = [], []
    for batch, wps, labels in tqdm(DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer), shuffle=False)):
        batch = {k: v.to(gpu) for k, v in batch.items()}
        wps = wps.to(gpu)
        predicted_scores += torch.clamp(model(batch, wps), 0, 1).flatten().tolist()
        actual_scores += labels
    predicted_scores = np.array(predicted_scores)
    actual_scores = np.array(actual_scores)
    mse = np.square(np.subtract(predicted_scores, actual_scores)).mean()
    model.train()
    return pearsonr(predicted_scores, actual_scores)[0], mse

global_steps = 0
best_eval = 0
for epoch in range(10):
    print("Epoch ", epoch)
    total_loss = 0
    total = 0
    for batch, wps, labels in DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer), shuffle=True):
        batch = {k: v.to(gpu) for k, v in batch.items()}
        wps = wps.to(gpu)
        labels = torch.tensor(labels).to(gpu)
        outputs = model(batch, wps).squeeze()

        loss = loss_fn(outputs.squeeze(), labels)
        cur_batch_size = labels.size(0)
        total_loss += loss.item() * cur_batch_size
        total += cur_batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1

        if global_steps % 100 == 0:
            eval_result, dev_loss =  eval(dev_dataset)
            if eval_result > best_eval:
                best_eval = eval_result
            print("Epoch %s Global steps: %s Train loss: %.4f Dev loss: %.4f Current r:%.4f Best r: %.4f" % (epoch, global_steps, total_loss/total, dev_loss, eval_result, best_eval))
