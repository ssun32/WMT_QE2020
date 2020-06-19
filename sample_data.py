from glob import glob
import random
random.seed(12345)

lcodes = [("en","de"),("en","zh"),("ro","en"),("et","en"),("si","en"),("ne","en"), ("ru", "en")]

for src, tgt in lcodes:
    print(src, tgt)
    train_f = "data/%s-%s/train.%s%s.df.short.tsv"%(src,tgt,src,tgt)
    word_prob_f = "data/%s-%s/word-probas/word_probas.train.%s%s"%(src,tgt,src,tgt)
    mt_f = "data/%s-%s/word-probas/mt.train.%s%s"%(src,tgt,src,tgt)

    train,train_wp,train_mt=[],[],[]
    for i, l in enumerate(open(train_f)):
        if i == 0:
            header = l
        else:
            train.append(l)
    for l in open(word_prob_f):
        train_wp.append(l)
    for l in open(mt_f):
        train_mt.append(l)

    all_samples = list(zip(train, train_wp, train_mt))

    train_samples =  all_samples[:6000]
    traindev_samples = all_samples[6000:]
    print(len(traindev_samples))

    with open(train_f.replace("train", "train"), "w") as ftrain, \
         open(word_prob_f.replace("train", "train"), "w") as fwp, \
         open(mt_f.replace("train", "train"), "w") as fmt:
             print(header, end='', file=ftrain)

             for t,w,m in train_samples:
                 print(t, end='', file=ftrain)
                 print(w, end='', file=fwp)
                 print(m, end='', file=fmt)

    with open(train_f.replace("train", "traindev"), "w") as ftrain, \
        open(word_prob_f.replace("train", "traindev"), "w") as fwp, \
        open(mt_f.replace("train", "traindev"), "w") as fmt:
            print(header, end='', file=ftrain)

            for t,w,m in traindev_samples:
                print(t, end='', file=ftrain)
                print(w, end='', file=fwp)
                print(m, end='', file=fmt)


    for p in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
        samples = random.sample(train_samples, int(len(train_samples) * p))

        with open(train_f.replace("train", "train_%s"%p), "w") as ftrain, \
             open(word_prob_f.replace("train", "train_%s"%p), "w") as fwp, \
             open(mt_f.replace("train", "train_%s"%p), "w") as fmt:
                 print(header, end='', file=ftrain)

                 for t,w,m in samples * int(1/p):
                     print(t, end='', file=ftrain)
                     print(w, end='', file=fwp)
                     print(m, end='', file=fmt)



