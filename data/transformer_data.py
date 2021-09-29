# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable

def nopeak_mask(n):
    np_mask = np.triu(np.ones((1, n, n), np.uint8), k=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask


def create_masks(text, summ, pad):
    text_mask = (text != pad).unsqueeze(-2)

    if summ is not None:
        summ_mask = (summ != pad).unsqueeze(-2)
        n = summ.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(n)
        summ_mask = summ_mask & np_mask
        pass
    else:
        summ_mask = None
        pass
    return text_mask, summ_mask
