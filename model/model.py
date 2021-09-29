# -*- coding: utf-8 -*-

import os, sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from .gnn import GNN_Encoder
from .transformer import Encoder, Decoder
from misc import whoami

logger = logging.getLogger(__name__)


class GNN_Transformer(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()
        vocab_size = config.lm.embeddings.word_embeddings.num_embeddings
        d_model = config.lm.embeddings.word_embeddings.embedding_dim

        self.gnn_encoder = GNN_Encoder(config) # 1st encoder: GNN Encoder

        if config.use_transformer_encoder:
            self.tns_encoder = Encoder(config)     # 2nd encoder: Transformer Encoder
            pass
        else:
            self.tns_encoder = None
            pass

        #self.decoder = Decoder(trg_vocab_size, d_model, N, n_heads, dropout)
        self.tns_decoder = Decoder(config)     # deocoder   : Transformer Decoder
        self.out = nn.Linear(d_model, vocab_size)
        pass

    def forward(self, *batch):
        T, TN, B, BN, HM, S_, SM, V, A1, A2 = batch

        g_outputs = self.gnn_encoder(T, TN, B, BN, V, A1, A2)
        #logger.info(f'{__class__.__name__}.forward: g_outputf.size:{g_outputs.size()}')
        if self.tns_encoder:
            e_outputs = self.tns_encoder(g_outputs, HM)
            #logger.info(f'{__class__.__name__}.forward: e_outputf.size:{e_outputs.size()}')
            pass
        else:
            e_outputs = g_outputs
            pass

        d_outputs = self.tns_decoder(S_, e_outputs, HM, SM)
        #logger.info(f'{__class__.__name__}.forward: d_outputf.size:{d_outputs.size()}')
        output = self.out(d_outputs)
        return output

    #def get_word_embedding(self, word):
    #    return self.gnn_encoder.get_word_embedding(word)

    pass


def get_model(config):
    model = GNN_Transformer(config)

    if not config.no_cuda:
        model.to(config.device)
        pass

    return model

    #if 1 < torch.cuda.device_count():
    #    logger.info(f"Let's use {torch.cuda.device_count()} GPUs...")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    model = nn.DataParallel(model)
    #    pass

    #if not config.no_cuda:
    #    model.to(config.device)
    #    pass
    #return model


def main():
    pass


if __name__ == '__main__':
    main()
    pass
