# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import copy
import logging

from misc import whoami

logger = logging.getLogger(__name__)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()

        self.size = d_model
        self.eps = eps  # epsilon = 1e-6

        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        pass

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

    pass


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        pass

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
        pass

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        '''
        args:
          n_heads: number of heads in transformer
          d_model: depth of model
        '''
        super(MultiHeadAttention, self).__init__()

        # In the case of the Encoder, V, K and Q will simply be identical copies of the embedding vector
        # (plus positional encoding). They will have the dimensions Batch_size * seq_len * d_model.
        # In multi-head attention we split the embedding vector into N heads, so they will then have the
        # dimensions batch_size * N * seq_len * (d_model / N).
        # This final dimension (d_model / N ) we will refer to as d_k.

        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        pass

    def forward(self, q, k, v, mask=None):
        # dimension of q, k, v: (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        v = self.k_linear(v).view(batch_size, -1, self.h, self.d_k)

        # transpose to get dimensions (batch_size, n_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view( batch_size
                                                         , -1
                                                         , self.d_model
                                                         )
        output = self.out(concat)

        return output
    pass


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        # set d_ff as a default to 2048
        self.fc1 = nn.Linear(d_model, d_ff) # d_ff: dimension of feed forwarding layer
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        pass

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    pass


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        pass

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    pass


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(n_heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(n_heads, d_model, dropout=dropout)

        self.ff = FeedForward(d_model, dropout=dropout)
        pass

    def forward(self, x, e_outputs, src_mask, trg_mask):
        '''
        * Params:
          x        : target data
          e_outputs: encoder-output
          src_mask : mask for source(encoder-output)
          trg_mask : mask for target(decoder-input)
        '''
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)

        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)

        x = x + self.dropout_3(self.ff(x2))

        return x
    pass


class PositionalEncoder(nn.Module):
    def __init__(self, device, d_model, max_seq_len=200, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        fnm = whoami()

        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        logger.info(f'{__class__.__name__}.{fnm}: d_model:{d_model} / max_seq_len:{max_seq_len}')

        # create constant pe(positional Encoder) matrix with
        # values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            # d_model assumed to be even number.
            for i in range(0, d_model, 2):
                pe[pos, i  ] = math.sin(pos / (10000**((2 *  i   )/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000**((2 * (i+1))/d_model)))
                pass
            pass
        pe = pe.unsqueeze(0) # 최상위 차원추가
        self.register_buffer('pe', pe)
        self.device = device
        pass

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)

        # add constant to embedding
        seq_len = x.size(1) # batch * seq_len * emb_size --> seq_len
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        x = x + (pe.to(self.device) if x.is_cuda else pe)
        return self.dropout(x)
    pass


class Encoder(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()
        fnm = whoami()

        N = config.n_tns_layers  # number of transformer layers
        vocab_size = config.lm.embeddings.word_embeddings.num_embeddings
        d_model = config.lm.embeddings.word_embeddings.embedding_dim
        n_heads = config.n_heads

        assert d_model % n_heads == 0, f'd_model({d_model}) must be divided by n_heads({n_heads})'

        dropout = config.dropout_rate 
        logger.info(f'{fnm}: N:{N} / vocab_size:{vocab_size} / d_model:{d_model} / n_heads:{n_heads}')

        self.N = N   # number of EncoderLayer
        #self.embed = Embedder(vocab_size, d_model)  # embedding은 bert embedding 사용

        self.pe = PositionalEncoder(config.device, d_model, max_seq_len=int(1.5 * (config.MAX_TEXT_TOKENS+config.MAX_SENTS)), dropout=dropout)

        logger.info(f'{fnm}: PositionalEncoder module created')  # YOYO
        self.layers = get_clones(EncoderLayer(d_model, n_heads, dropout), N)
        self.norm = Norm(d_model)

        logger.info(f'{fnm}: Transformer Encoder created')

    def forward(self, src, mask):
        #x = self.embed(src)
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
            pass
        return self.norm(x)
    pass


class Decoder(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()
        fnm = whoami()
        pass

    def forward(self, batch):
        T, B, A1, A2, VT, VB = batch
        pass
    pass


class Decoder(nn.Module):
    def __init__(self, config):
        super(__class__, self).__init__()
        N = config.n_tns_layers  # number of transformer layers
        vocab_size = config.lm.embeddings.word_embeddings.num_embeddings
        d_model = config.lm.embeddings.word_embeddings.embedding_dim
        n_heads = config.n_heads
        dropout = config.dropout_rate

        self.N = N  # numbef DecoderLayer
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(config.device, d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, n_heads, dropout), N)
        self.norm = Norm(d_model)
        pass

    def forward(self, summ, e_outputs, text_mask, summ_mask):
        x = self.embed(summ)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, text_mask, summ_mask)
            pass
        return self.norm(x)
    pass


