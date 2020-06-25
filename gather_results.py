from glob import glob 
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr

lcodes = [("en","de"), ("en","zh"), ("et","en"), ("ro","en"), ("si", "en"), ("ne", "en"), ("ru", "en")]
gold_labels = defaultdict(list)
#get gold labels
for f in glob("data/*/dev.*.df.short.tsv"):
    ld = f.split("/")[-1].split(".")[1]
    src = ld[:2]
    tgt = ld[2:]

    for i, l in enumerate(open(f)):
        if i > 0:
            gold_labels[(src, tgt)].append(float(l.split("\t")[-2]))

#H1 and H1.1
results = {}
exp_names = []
for f in tqdm(glob("experiments/H1/*/run*/*test*")):
    if "H1.2" in f or "H1.05" in f:
        continue
    if "fewshot" in f:
        continue
    _, _, exp_name, run, score_f = f.split("/")
    src, tgt = score_f.split(".")[0].split("_")
    scores = [float(s) for s in open(f)]

    pc = pearsonr(gold_labels[(src, tgt)], scores)[0]
    
    mlp = "lang_spec" if "lang_spec" in score_f else "lang_agnost"
    exp_names.append(exp_name)

    k = (exp_name, (src, tgt), mlp)
    if k not in results:
        results[k] = []
    results[k].append(pc)

exp_names = set(exp_names)

header = ["Exp_name"] + ["-".join(ld) for ld in lcodes]
print("&".join(header)+"\\\\")
for exp_name in sorted(exp_names, key=lambda x: (1 if "mtl" in x else 0, x)):
    for mlp in ["lang_spec", "lang_agnost"]:
        if mlp == "lang_spec" and "mtl" not in exp_name:
            continue
        tmp_results = []
        for ld in lcodes:

            k = (exp_name, ld, mlp)
            if k in results:
                print(k, results[k])
                mean_result = np.mean(results[k])
                std = np.std(results[k])
                tmp_results.append("%.3f/%.3f"%(mean_result,std))
                #tmp_results.append("%.4f/%.4f/%.4f"%(mean_result, min(results[k]), max(results[k])))
            else:
                tmp_results.append("-")

        if "mtl" not in mlp:
            mlp = "-"
        print("%s&%s\\\\"%(exp_name, "&".join(tmp_results)))

print()

"""
#H1.2
#zero shot and few shot
results = {}
for f in glob("experiments_xlmr/H1/*/run*/*test*"):
    if "fewshot" in f:
        continue
    _, _, exp_name, run, score_f = f.split("/")
    src, tgt = score_f.split(".")[0].split("_")
    scores = [float(s) for s in open(f)]

    pc = pearsonr(gold_labels[(src, tgt)], scores)[0]
   
    shot = float(exp_name.split("_")[-2])
    mlp = "Base"

    k = ((src, tgt), shot, mlp)
    if k not in results:
        results[k] = []
    results[k].append(pc)

for f in glob("experiments_xlmr/H1.2/*/run*/*test*"):
    _, _, exp_name, run, score_f = f.split("/")
    src, tgt = score_f.split(".")[0].split("_")
    scores = [float(s) for s in open(f)]

    pc = pearsonr(gold_labels[(src, tgt)], scores)[0]
    
    shot = 0 if "0shot" in exp_name else float(exp_name.split("_")[-2])
    mlp = "LS MLP" if "lang_spec" in score_f else "LA MLP"

    k = ((src, tgt), shot, mlp)
    if k not in results:
        results[k] = []
    results[k].append(pc)

def upper(ld):
    return (ld[0][0].upper() + ld[0][1], ld[1][0].upper()+ld[1][1])

header = ["LD", "TYPE"] + [("-".join(upper(ld))) for ld in lcodes]
print("&".join(["\\textbf{%s}"%h for h in header])+"\\\\")
for shot in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
    for mlp in ["Base", "LS MLP", "LA MLP"]:
        tmp_results = []

        for ld in lcodes:

            k = (ld, shot, mlp)
            if k in results:
                mean_result = sum(results[k])/len(results[k])
                tmp_results.append("%.3f"%mean_result)
                #tmp_results.append("%.4f/%.4f/%.4f"%(mean_result, min(results[k]), max(results[k])))

        ld_name = "\multirow{3}{*}{\\textbf{%s}}"%(shot) if mlp == "Base" else ""
        print("%s&%s&%s\\\\"%(ld_name, mlp.replace("_","-"), "&".join(tmp_results)))
        if mlp != "LA MLP":
            print("\cmidrule{2-9}")
        else:
            print("\midrule")
"""
