from PIL import Image
import json
import os
import numpy as np
# info = "../ADEChallengeData2016/objectInfo150.txt"
# with open(info, 'r') as info_f:
#     _ = info_f.readline()
#     ratios = [float(line.split()[1].strip()) for line in info_f.readlines()]
# print(ratios)
# odgt = "data/training.odgt"
# list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
# label_dist = np.zeros(150)
# for sample in list_sample:
#     fpath_segm = os.path.join("..", sample["fpath_segm"])
#     segm = Image.open(fpath_segm)
#     segm = np.array(segm).flatten() - 1
#     for i in range(150):
#         label_dist[i] += np.sum(segm==i)
# label_dist = label_dist/np.sum(label_dist)
# print(label_dist)
# np.save("data/label_bias.npy", label_dist)

odgt = "data/training.odgt"
list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]
train_size = len(list_sample)
subset = np.random.choice(np.arange(train_size), 1000, False)
list_sample_1k = [list_sample[i] for i in subset]
with open("data/training_1k.odgt", "w") as f:
    for sample in list_sample_1k:
        json.dump(sample, f)
        f.write("\n")