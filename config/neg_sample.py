from curses import tparm
import torch
import numpy as np
import os
import random


def load_samples(address):
    address = os.path.join(address, 'negetive_sample.txt')
    # address = os.path.join(address, 'train2id.txt')
    sampled_triples = {}
    with open(address) as f:
        # f.readline()
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            line = [int(ele) for ele in line]
            # print(line)
            if line[0] not in sampled_triples:
                sampled_triples[line[0]] = [line[2], line[1]]
            else:
                sampled_triples[line[0]] += [line[1]]
    for element in sampled_triples.keys():
        relation = sampled_triples[element][0]
        tail = sampled_triples[element][1:]
        sampled_triples[element] = [relation, tail]
    # print(sampled_triples)
    return sampled_triples


def samples(heads, tails, relations, labels, sampled_triples):
    # 输入是np 不是list
    # print(relations)
    
    triple_batch = [heads, tails, relations, labels]
    triple_batch = np.array(triple_batch, dtype=np.int32).T
    triple_batch = triple_batch.tolist()
    # print('before')
    # print([len(triple_batch),len(triple_batch[0])])

    # print(triple_batch)
    # print(len(triple_batch))
    length = len(triple_batch)//2

    positive_triple = triple_batch[:length]
    negitive_triple = triple_batch[length:]
    # print([len(positive_triple),len(positive_triple[0])])
    # print([len(negitive_triple),len(negitive_triple[0])])

    # print(len(positive_triple))
    # print(len(negitive_triple))
    # print()
    for triple in negitive_triple:
        # print(triple[0])
        if triple[0] in sampled_triples:
            # print([triple[0], sampled_triples[triple[0]]])
            triple[1] = int(random.choice(sampled_triples[triple[0]][1]))
            # print(triple[1])
    triple_batch = positive_triple + negitive_triple
    triple_batch = np.array(triple_batch, dtype=np.int64)

    # print(triple_batch)
    # print('after')
    # print([len(triple_batch), len(triple_batch[0])])

    # print('\n tail')
    # print(tails)
    # print(len(tails))
    # print(triple_batch[:, 1])
    # print(len(triple_batch[:, 1]))


    # print('\n relation')
    # print(relations)
    # print(len(relations))
    # print(triple_batch[:, 2])
    # print(len(triple_batch[:, 2]))

    # print('h t r label size')
    # print([len(sampled_head)])
    # print([len(sampled_head), len(sampled_head[0])])
    # print([len(sampled_tail), len(sampled_tail[0])])
    # print([len(sampled_relation), len(sampled_relation[0])])
    # print([len(sampled_label), len(sampled_label[0])])
    return triple_batch[:,1]


if __name__ == '__main__':
    pass
