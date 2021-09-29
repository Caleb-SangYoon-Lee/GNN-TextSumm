# -*- coding: utf-8 -*-

import os
import glob
import pickle
import numpy as np

import logging

import torch
from torch.utils.data import Dataset, Sampler

from misc import whoami, split_by_sep

logger = logging.getLogger(__name__)




def normal_dist(x, m, s):
    # s: std (standard deviation)
    # m: mean
    c = 1 / (((2 * np.pi ) ** 0.5) * s)
    z = -((x - m) ** 2) / (2 * (s ** 2))
    p = c * np.exp(z)
    return p


def get_adjacent_matrix_from_sents(n_sents, d=5):
    n = n_sents

    adj = np.zeros((n, n))

    for i in range(n):
        lb = i - d       # left bound
        if lb < 0:
            lb = 0
            pass
        rb = i + (d + 1) # right bound
        if n < rb:
            rb = n
            pass

        for j in range(lb, rb):
            adj[i, j] = 1
            pass
        pass
    return adj



def ids2adj_mat_without_padding(n):
    adj = np.eye(n)
    d = int(n * 0.05)

    if 25 < d:
        d = 25
        pass
    elif d < 5:
        d = 5
        pass

    for i in range(n):
        lb = i - d       # left bound
        if lb < 0:
            lb = 0
            pass
        rb = i + (d + 1) # right bound
        if n < rb:
            rb = n
            pass
        for j in range(lb, rb):
            adj[i, j] = 1
            pass
        pass
    return adj





class GNN_Dataset(Dataset):
    def __init__(self, config, mode, data):
        fnm = f'{__class__.__name__}.{whoami()}'
        self.mode = mode
        self.text_list = data['text']
        self.summ_list = data['summ']
        self.best_id_list = data['best_id']
        self.n_data = len(self.text_list)

        logger.info(f'{fnm}: mode:{mode}: n_data({self.n_data}) are available')

        n_max_sent_dist = 250 * 2
        self.sent_dists = self.get_sent_dists(n_max_sent_dist)

        self.pad_token_id = config.tokenizer.pad_token_id
        self.sep_token_id = config.tokenizer.sep_token_id

        self.MAX_TEXT_TOKENS = config.MAX_TEXT_TOKENS
        self.MAX_SUMM_TOKENS = config.MAX_SUMM_TOKENS
        self.MAX_SENTS = config.MAX_SENTS
        pass

    def get_sent_dists(self, n_max_sent_dist):
        sent_dists = np.zeros(n_max_sent_dist)

        m = 1.0
        s = 8.0

        for i in range(n_max_sent_dist):
            sent_dists[i] = normal_dist(-i, m, s) * 100
            pass
        return sent_dists

    def __len__(self):
        return self.n_data

    def get_sent_dist_mat(self, ids_len_list, n_ids, n_sents):
        dist_mat = np.zeros((n_ids, n_sents))

        pos = 0
        for i, n in enumerate(ids_len_list):
            dist_mat[pos:pos+n,:] = self.sent_dists[i:i+n_sents]
            pos += n
            pass
        return dist_mat

    def restrain_text_ids(self, text_ids, n_sents):
        ids_len_list = list()
        ids_list = list()

        MAX_TEXT_TOKENS = self.MAX_TEXT_TOKENS
        MAX_SENTS = self.MAX_SENTS

        for i, ids in enumerate(text_ids):
            if (MAX_TEXT_TOKENS + MAX_SENTS) <= len(ids_len_list) + len(ids_list):
                break

            n_ids_curr = len(ids_list)
            n_ids = len(ids)

            if i < n_sents - 1:
                if MAX_TEXT_TOKENS <= n_ids_curr + 1:
                    break
                ids += [self.sep_token_id]
                n_ids += 1
                pass

            if MAX_TEXT_TOKENS < n_ids_curr + n_ids:
                n_clipped = n_ids_curr + n_ids - MAX_TEXT_TOKENS
                if 0 < n_clipped:
                    ids = ids[:-n_clipped]

                    ids_len_list.append(len(ids))
                    ids_list.extend(ids)
                    break
                pass
            else:
                ids_len_list.append(len(ids))
                ids_list.extend(ids)
                pass
            pass
        return ids_len_list, ids_list


    def __getitem__(self, index):
        text_ids = self.text_list[index]
        summ_ids = self.summ_list[index]
        best_ids = self.best_id_list[index]
        
        n_sents = len(text_ids)

        MAX_SUMM_TOKENS = self.MAX_SUMM_TOKENS

        ids_len_list, ids_list = self.restrain_text_ids(text_ids, n_sents)

        n_ids = len(ids_list) # total ids for the text

        n_summ = len(summ_ids)

        if MAX_SUMM_TOKENS < n_summ:
            summ_ids = summ_ids[:MAX_SUMM_TOKENS]
            pass

        # prepare text related graph
        n1 = n_ids
        c1 = ids_list
        adj1 = ids2adj_mat_without_padding(n1)  # adj1: (n1, n1) matrix

        # prepare super dummy node graph
        n2 = len(best_ids)
        c2 = best_ids
        adj2 = get_adjacent_matrix_from_sents(n2) # adj2: (n2, n2) matrix

        # aggregrate
        agg_adj1 = np.zeros((n1+n2, n1+n2))

        agg_adj1[:n1, :n1] = adj1
        agg_adj1[n1:, n1:] = adj2

        agg_adj2 = np.copy(agg_adj1)
        dm = self.get_sent_dist_mat(ids_len_list, n_ids, n_sents) # distance from sentence to super-dummy-nodes

        agg_adj2[:n1,n1:] = np.copy(dm)
        agg_adj2[n1:,:n1] = np.copy(np.transpose(dm))

        # node indice for aggregation
        valid = np.zeros((n1 + n2,))
        valid[:n1] = 1

        text_dict = {
                'T' : c1,
                'TN': n_ids,
                'B' : best_ids,
                'BN': n_sents,
                'V' : valid,
                'S' : summ_ids,
                'A1': agg_adj1,
                'A2': agg_adj2,
                }
        return text_dict
    pass



class Collator:
    def __init__(self, config):
        self.MAX_SUMM_TOKENS = config.MAX_SUMM_TOKENS
        pass

    def __call__(self, batch):
        MAX_SUMM_TOKENS = self.MAX_SUMM_TOKENS

        max_ids  = max([data['TN'] for data in batch if data is not None])
        max_sent = max([data['BN'] for data in batch if data is not None])
        max_elm  = max_ids + max_sent

        max_summ = max([len(data['S']) for data in batch if data is not None])

        n_batch = len(batch)

        T  = np.zeros((n_batch, max_ids ), dtype=np.int) # texts
        TN = np.zeros((n_batch, ), dtype=np.int)

        B  = np.zeros((n_batch, max_sent), dtype=np.int)
        BN = np.zeros((n_batch, ), dtype=np.int)

        V  = np.zeros((n_batch, max_elm))

        S  = np.zeros((n_batch, MAX_SUMM_TOKENS), dtype=np.int)  # summaries
        SM = np.zeros((n_batch, max_summ), dtype=np.int)

        A1 = np.zeros((n_batch, max_elm, max_elm)) # adjacent matrix A1
        A2 = np.zeros((n_batch, max_elm, max_elm)) # adjacent matrix A2

        for i in range(n_batch):
            data = batch[i]

            # process text ids
            ids = data['T']
            n_ids = data['TN']

            T [i,:n_ids] = np.array(ids, np.int)
            TN[i] = n_ids

            # process best ids
            best_ids = data['B']
            n_sent = data['BN']

            B [i,:n_sent] = np.array(best_ids, np.int64)
            BN[i] = n_sent

            n_elm = n_ids + n_sent

            V[i,:n_elm] = batch[i]['V']

            # process summ. ids
            summ = data['S']
            n_summ = len(summ)

            S [i,:n_summ] = np.array(summ, np.int)
            SM[i,:n_summ - 1] = 1 # valid mask of summary (last appended sep excluded)

            # adj1, adj2: adjascent matrix

            A1[i, :n_elm, :n_elm] = data['A1']
            A2[i, :n_elm, :n_elm] = data['A2']

            pass

        T  = torch.from_numpy(T)
        B  = torch.from_numpy(B)
        V = torch.from_numpy(V).float()

        S  = torch.from_numpy(S)

        A1 = torch.from_numpy(A1).float()
        A2 = torch.from_numpy(A2).float()

        return T, TN, B, BN, V, S, A1, A2
    pass


class BatchSampler(Sampler):
    def __init__(self, batch_size, n_data):
        #super(BatchSampler, self).__init__()
        self.batch_size = batch_size
        self.n_data = n_data
        self.n_batches = n_data // batch_size
        pass

    def __iter__(self):
        random_batches = torch.randperm(self.n_batches)
        for i in random_batches:
            for j in range(self.batch_size):
                yield i * self.batch_size + j
                pass
            pass
        pass

    def __len__(self):
        return self.n_data

    pass


def main():
    pass


if __name__ == '__main__':
    main()
    pass
