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

        if not config.tns_only:
            self.gnn_encoder = GNN_Encoder(config) # 1st encoder: GNN Encoder
            pass
        else:
            self.gnn_encoder = None # 1st encoder: GNN Encoder
            pass

        if config.use_transformer_encoder:
            self.tns_encoder = Encoder(config)     # 2nd encoder: Transformer Encoder
            pass
        else:
            self.tns_encoder = None
            pass

        self.tns_decoder = Decoder(config)     # deocoder   : Transformer Decoder
        self.out = nn.Linear(d_model, vocab_size)
        pass

    def forward(self, *batch):
        if self.gnn_encoder:
            T, TN, B, BN, HM, S_, SM, V, A1, A2 = batch
            outputs = self.gnn_encoder(T, TN, B, BN, V, A1, A2)
            if self.tns_encoder:
                outputs = self.tns_encoder(outputs, HM)
                pass
            pass
        else:
            T, TN, HM, S_, SM, V = batch
            outputs = self.tns_encoder(T, HM)
            pass

        outputs = self.tns_decoder(S_, outputs, HM, SM)
        outputs = self.out(outputs)
        return outputs

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
