import os
import json
from glob import glob
n_runs=5
batch_size_per_gpu=8
 
ids = {
 "all":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")],
 "sharing_src":[("en","de"), ("en","zh")],
 "sharing_tgt":[("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")],
 "random2":[("en","zh"),("ro","en")],
 "0shot_no_ende":[("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_enzh":[("en","de"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_roen":[("en","de"),("en","zh"),("et","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_eten":[("en","de"),("en","zh"),("ro","en"),("si","en"),("ne","en"), ("ru","en")],
 "0shot_no_sien":[("en","de"),("en","zh"),("ro","en"),("et","en"),("ne","en"), ("ru","en")],
 "0shot_no_neen":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"), ("ru","en")],
 "0shot_no_ruen":[("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"), ("ne","en")],
 "en_de":[("en","de")],
 "en_zh":[("en","zh")],
 "ro_en":[("ro","en")],
 "et_en":[("et","en")],
 "si_en":[("si","en")],
 "ne_en":[("ne","en")],
 "ru_en":[("ru","en")]
 }


#H1 experiments
def get_files(lcodes, split="train", sample_dict={}):
    tsv_file, mt_file, wp_file = [], [], []
    for src_lcode, tgt_lcode in lcodes:
        if split=="train" and (src_lcode, tgt_lcode) in sample_dict:
            p = sample_dict[(src_lcode, tgt_lcode)]
            cur_split = "train_%s"%p
        elif split=="dev":
            cur_split="traindev"
        elif split=="test":
            cur_split="dev"
        else:
            cur_split=split
        filedir = "data/%s-%s"%(src_lcode, tgt_lcode)
        tsv_file += glob("%s/%s.*.tsv" % (filedir,cur_split))
        mt_file += glob("%s/word-probas/mt.%s.*" % (filedir,cur_split))
        wp_file += glob("%s/word-probas/word_probas.%s.*" % (filedir,cur_split))
    return tsv_file, mt_file, wp_file

def make_config(train, 
                dev, 
                test, 
                exp="H1",
                name="all",
                n_runs=1, 
                batch_size_per_gpu=8,
                sample_dict={}):
    for run in range(n_runs):
        config = {
                  "model_name":"xlm-roberta-large",
                  "model_dim":1024,
                  "epochs":20,
                  #"model_name":"bert-base-multilingual-cased",
                  #"model_dim":768,
                  "learning_rate":1e-6,
                  "batch_size_per_gpu": batch_size_per_gpu,
                  "accum_grad": 1,
                  "eval_interval": 100 * len(ids[train[0]]),
                  "use_word_probs":False,
                  "use_secondary_loss":False,
                  "train":[],
                  "dev":[],
                  "test":[]
                }
        output_dir = "experiments/%s/%s/run%s/" % (exp, name, run)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w") as fout:
            config["output_dir"] =  output_dir

            for split_name, split in [("train", train), ("dev", dev), ("test", test)]:
                for id in split:
                    tsv_file, mt_file, wp_file = get_files(ids[id], split=split_name, sample_dict=sample_dict)
                    if len(tsv_file) > 1:
                        id = "all"
                    config[split_name].append({"id": id, "tsv_file":tsv_file, "mt_file":mt_file, "wp_file":wp_file})
            print(json.dumps(config, indent=4), file=fout)


#H1 experiments

#single languages
all_lds = ["_".join(ld) for ld in ids["all"]]
for ld in all_lds:
    make_config([ld], [ld], [ld], exp="H1", name=ld, n_runs=5, batch_size_per_gpu=8)

#train on all LD, test on single languages
make_config(["all"], all_lds, all_lds, exp="H1", name="all", n_runs=5, batch_size_per_gpu=8)

#MTL
make_config(["all"] + all_lds, all_lds, all_lds, exp="H1", name="mtl", n_runs=5, batch_size_per_gpu=8)


#H1.1 experiments
sharing_src = ["_".join(ld) for ld in ids["sharing_src"]]
sharing_tgt = ["_".join(ld) for ld in ids["sharing_tgt"]]
random2 = ["_".join(ld) for ld in ids["random2"]]
make_config(["sharing_src"] + sharing_src, sharing_src, sharing_src, exp="H1.1", name="mtl_sharing_src", n_runs=n_runs, batch_size_per_gpu=batch_size_per_gpu)
make_config(["sharing_tgt"] + sharing_tgt, sharing_tgt, sharing_tgt, exp="H1.1", name="mtl_sharing_tgt", n_runs=n_runs, batch_size_per_gpu=batch_size_per_gpu)
make_config(["random2"] + random2, random2, random2, exp="H1.1", name="mtl_random2", n_runs=n_runs, batch_size_per_gpu=batch_size_per_gpu)

#H1.2 experiments
#zero shot
for ld in ids["all"]:
    id = "0shot_no_%s%s" % ld
    tmp = ["_".join(ld) for ld in ids[id]]
    make_config([id] + tmp, all_lds, ["%s_%s"%ld], exp="H1.2", name="mtl_%s"%id, n_runs=n_runs, batch_size_per_gpu=batch_size_per_gpu)

#few shot
for ld in ids["all"]:
    for p in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        id = "fewshot_%s_%s%s" % (p, ld[0], ld[1])
        make_config(["all"] + all_lds, all_lds, ["%s_%s"%ld], exp="H1.2", name="mtl_%s"%id, n_runs=n_runs, batch_size_per_gpu=batch_size_per_gpu, sample_dict={ld:p})


