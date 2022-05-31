import torch
import numpy as np

def discounted_cum_sum(seq, discount):
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq