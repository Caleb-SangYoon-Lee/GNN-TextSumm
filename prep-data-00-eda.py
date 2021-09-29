# -*- coding: utf-8 -*-

import os, sys
import glob
import re
import pandas as pd
import pickle

import argparse
import logging

import matplotlib.pyplot as plt

from pykospacing import Spacing
from kss import split_sentences

from misc import whoami, run_scandir, run_scandir_re
from misc import init_logger, set_seed, load_config

from data import get_data_from_ai_hub, MAX_SUMM_TOKENS


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')

    parser.add_argument('--config-dir' , type=str,  default=os.path.join('.' 'config'), help='config. directory')
    parser.add_argument('--config-file', type=str,  required=True  , help='config. file in config. directory')
    parser.add_argument('--mode'       , type=str,  default='train', help='train or test')
    parser.add_argument('--show-hist'  , type=bool, default=False , help='to show sentence lengths distribution')

    return parser.parse_args()


def main():
    fnm = whoami()

    init_logger()

    # parse command-line argments 
    args = get_args()
    logger.info(f'{fnm}: args:\n{args}')

    # get config.
    config_file_path = os.path.join(args.config_dir, args.config_file)
    config = load_config(config_file_path)

    logger.info(f'{fnm}: config:\n{config}')


    # ------------------------------
    # get text, summarization stings
    # ------------------------------

    # init. text/summarization list
    text_list, summ_list = list(), list()

    # get data from AI Hub.
    text_list, summ_list = get_data_from_ai_hub(config, args.mode, text_list, summ_list)

    logger.info(f'#1: len(text_list):{len(text_list)}')
    logger.info(f'#1: len(summ_list):{len(summ_list)}')

    max_token_in_sent = 0
    max_token_in_text = 0
    max_token_in_summ = 0

    max_sent_in_text = 0
    max_sent_in_summ = 0


    text_token_len_list = list()
    summ_token_len_list = list()

    n_long_article = 0
    n_null_article_skipped = 0 # 2021-09-07: added

    spacer = Spacing()

    sent_token_len_list = list()
    sent_len_list = list()

    for i, (text, summ) in enumerate(zip(text_list, summ_list), 1):
        if not isinstance(text, list):
            text = split_sentences(text)
            pass

        if isinstance(summ, list):
            summ = ' '.join(summ).strip()
            pass
        else:
            summ = summ.strip()
            pass

        #######################################################

        # text 처리

        text_token_list = list()
        n_token_in_text = 0
        for sent in text:
            if not sent:
                continue

            tokens = sent.split()

            n_tokens = len(tokens)
            if max_token_in_sent < n_tokens:
                max_token_in_sent = n_tokens
                pass

            sent_token_len_list.append(n_tokens)
            n_token_in_text += n_tokens

            text_token_list.append(tokens)
            pass

        sent_len_list.append(len(sent_token_len_list))

        #######################################################

        # summ(요약) 처리

        summ_tokens = summ.split()

        #######################################################

        n_token_in_summ = len(summ_tokens)

        if max_token_in_text < n_token_in_text:
            max_token_in_text = n_token_in_text
            pass

        if max_token_in_summ < n_token_in_summ:
            max_token_in_summ = n_token_in_summ
            pass

        text_token_len_list.append(n_token_in_text)
        summ_token_len_list.append(n_token_in_summ)

        if summ == '' or len(text_token_list) == 0:
            n_null_article_skipped += 1
            continue

        if MAX_SUMM_TOKENS < n_token_in_summ + 1:
            n_long_article += 1
            pass

        if i % 500 == 0:
            logger.info(f'{i:>5} of {len(text_list):>5}: {i * 100 / len(text_list):5.2f}%')
            pass

        if i <= 10:
            logger.info(f'{i:>2}#: text:{text} --> {text_token_list}(len:{len(text_token_list)})')
            logger.info(f'{i:>2}#: summ:{summ} --> {summ_tokens}(len:{len(summ_tokens)})')
            pass

        pass


    logger.info(f'{fnm}: max_token_in_text:{max_token_in_text}')
    logger.info(f'{fnm}: max_token_in_sent:{max_token_in_sent}')
    logger.info(f'{fnm}: max_token_in_summ:{max_token_in_summ}')


    logger.info(f'{fnm}: n_long_article:{n_long_article}')
    logger.info(f'{fnm}: n_null_article_skipped:{n_null_article_skipped}')

    logger.info('#' * 80)

    # ---------------------------------------
    # anal. distributions of length of tokens
    # ---------------------------------------
    length_dict = {'text':text_token_len_list, 'summ':summ_token_len_list}
    length_df = pd.DataFrame(length_dict)

    logger.info(length_df.head())

    ############################################

    sent_length_dict = {'sent_length':sent_token_len_list}
    sent_length_df = pd.DataFrame(sent_length_dict)
    logger.info(f'{fnm}: sent_length_df:\n{sent_length_df.head()}')
    logger.info('-' * 30)
    logger.info(f'{fnm}: sent_length_df.describe():{sent_length_df.describe()}')
    logger.info('-' * 30)

    if args.show_hist:
        length_df.hist(bins = 30)
        plt.show()

        sent_length_df.hist(bins = 30)
        plt.show()

        plt.hist(text_token_len_list, bins=100, density=True, range=(0, 3000))
        plt.show()

        plt.hist(summ_token_len_list, bins=100, density=True, range=(0, 300))
        plt.show()
        pass

    pass


if __name__ == '__main__':
    main()
    pass
