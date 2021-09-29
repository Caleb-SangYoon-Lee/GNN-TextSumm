# -*- coding: utf-8 -*-

import os, sys
import glob
import re
from collections import Counter
import pandas as pd
import pickle
import numpy as np

import argparse
import logging

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from misc import whoami, run_scandir, run_scandir_re
from misc import init_logger, set_seed, load_config

from transformers import  ElectraConfig, ElectraTokenizer


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')

    parser.add_argument('--config-dir' , type=str, default=os.path.join('.' 'config'), help='config. directory')
    parser.add_argument('--config-file', type=str, required=True  , help='config. file in config. directory')
    parser.add_argument('--mode'       , type=str, default='train', help='train or test')
    parser.add_argument('--ins-sos'    , type=bool,default=True   , help='insert sos([CLS]) in beginning of summary')

    return parser.parse_args()


def get_ranks(graph, d=0.85):  # d = damping factor
    A = graph
    matrix_size = A.shape[0]

    for id in range(matrix_size):
        A[id, id] = 0  # diagonal 부분을 0으로
        link_sum = np.sum(A[:, id])  # A[:, id] = A[:][id]

        if link_sum != 0:
            A[:, id] /= link_sum
            pass

        A[:, id] *= -d
        A[id, id] = 1
        pass

    B = (1-d) * np.ones((matrix_size, 1))

    ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b
    return ranks


def get_best_tokens(text_tokens):
    word_vectorizer = TfidfVectorizer()

    sent_list = [' '.join(sent_tokens) for sent_tokens in text_tokens]

    words_vectorized = word_vectorizer.fit_transform(sent_list).toarray().astype(float)

    words_graph = np.dot(words_vectorized.T, words_vectorized)
    ranks = get_ranks(words_graph)

    vocab = word_vectorizer.vocabulary_
    idx2word = {vocab[word]: word for word in vocab}

    maxes = np.argsort(ranks, axis=0)

    n_words = len(idx2word)
    kw_list = [idx2word[maxes[i][0]] for i in range(n_words)[::-1]]

    best_tokens = list()
    for sent_tokens in text_tokens:
        best_token = None

        for token in sent_tokens:
            for kw in kw_list:
                if kw in token:
                    best_token = token
                    break
                pass
            pass

        if best_token is None:
            token_counter = Counter(sent_tokens)
            best_token = token_counter.most_common(1)[0]
            logger.info(f'best_token not found replaced to {best_token} in {sent_tokens} for text:{text_tokens}')
            pass
        best_tokens.append(best_token)
        pass
    return best_tokens


def main():
    fnm = whoami()

    init_logger()

    args = get_args()
    logger.info(f'{fnm}: args:\n{args}')

    config_file_path = os.path.join(args.config_dir, args.config_file)
    config = load_config(config_file_path)

    logger.info(f'{fnm}: config:\n{config}')


    nlp_model_type = config.model_type
    nlp_model_path = config.model_name_or_path
    do_lower_case  = config.do_lower_case

    logger.info(f'{fnm}: nlp_model_type:{nlp_model_type} / nlp_model_path:{nlp_model_path} / do_lower_case:{do_lower_case}')

    tokenizer = ElectraTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    logger.info(f'tokenizer({nlp_model_type}) with {config.model_name_or_path} / {config.do_lower_case} done')


    #####################################################################################

    tokens_pkl_file_name = f'tokens.{args.mode}.pkl'
    tokens_pkl_file_path = os.path.join(config.data_dir, tokens_pkl_file_name)

    if not os.path.exists(tokens_pkl_file_path):
        logger.info(f'{fnm}: token file({tokens_pkl_file_path}) not found')
        return

    logger.info(f'{fnm}: token file({tokens_pkl_file_path}) found')

    with open(tokens_pkl_file_path, 'rb') as f:
        token_dict = pickle.load(f)
        pass

    text_tokens_list = token_dict['text']
    summ_tokens_list = token_dict['summ']

    n_text_tokens_list = len(text_tokens_list)
    n_summ_tokens_list = len(summ_tokens_list)

    logger.info(f'{fnm}: n_text_tokens_list:{n_text_tokens_list}')
    logger.info(f'{fnm}: n_summ_tokens_list:{n_summ_tokens_list}')

    assert n_text_tokens_list == n_summ_tokens_list, \
           f'lengths are not valid. [text_tokens_list({n_text_tokens_list}) != summ_tokens_list({n_summ_tokens_list})]'

    n_tokens_list = n_text_tokens_list

    text_ids_list = list()
    summ_ids_list = list()
    best_ids_list = list()

    #sep_id = tokenizer.sep_token_id

    for i in range(n_tokens_list):
        text_tokens = text_tokens_list[i]
        summ_tokens = summ_tokens_list[i]

        # process text tokens
        text_ids = list()
        for sent_tokens in text_tokens:
            sent_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
            text_ids.append(sent_ids)
            pass

        # process summ. tokens
        summ_ids = tokenizer.convert_tokens_to_ids(summ_tokens)
        if args.ins_sos:
            summ_ids = [tokenizer.cls_token_id] + summ_ids
            pass

        # get best token for each sentence in text
        best_tokens = get_best_tokens(text_tokens)
        best_ids = tokenizer.convert_tokens_to_ids(best_tokens)
        if i < 10:
            logger.info(f'sample best_tokens: len:{len(best_tokens)} / {best_tokens}')
            logger.info(f'sample best_ids: len(best_ids) / {best_ids}')
            logger.info('-' * 30)
            pass

        text_ids_list.append(text_ids)
        summ_ids_list.append(summ_ids)
        best_ids_list.append(best_ids)

        if (i + 1) % 100 == 0:
            logger.info(f'{i+1:>5} of {n_tokens_list:>5}: {i * 100 / n_tokens_list:5.2f}%')
            pass

        #if 500 <= i:
        #   break
        pass

    indexes = sorted(list(range(n_tokens_list)), key=lambda index: len(text_ids_list[index]), reverse=True)

    logger.info(f'len(indexes):{len(indexes)} vs n_tokens_list:{n_tokens_list}')
    logger.info(f'indexes[:10]:{indexes[:10]}')

    sorted_text_ids_list = list()
    sorted_summ_ids_list = list()
    sorted_best_ids_list = list()

    for i, index in enumerate(indexes):
        if i < 100:
            logger.info(f'[{i:>6}]#: index:{index:>6}# --> len(text_ids_list[{index:>6}]):{len(text_ids_list[index])}')
            pass
        sorted_text_ids_list.append(text_ids_list[index])
        sorted_summ_ids_list.append(summ_ids_list[index])
        sorted_best_ids_list.append(best_ids_list[index])
        pass

    ids_pkl_file_name = f'ids.{args.mode}.pkl'
    ids_pkl_file_path = os.path.join(config.data_dir, ids_pkl_file_name)

    ids_dict = {'text':sorted_text_ids_list, 'summ':sorted_summ_ids_list, 'best_id':sorted_best_ids_list}
    with open(ids_pkl_file_path, 'wb') as f:
        pickle.dump(ids_dict, f)
        pass

    logger.info(f'{fnm}: dump ids to {ids_pkl_file_path} done')
    logger.info('#' * 80)

    pass


if __name__ == '__main__':
    main()
    pass
