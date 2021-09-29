# -*- coding: utf-8 -*-

import os, sys
import glob
import json

import logging

from misc import whoami


# MAX_TEXT_TOKENS, MAX_SUMM_TOKENS, MAX_SENT_TOKENS, MAX_SENTS 등은 모두 config 파일로 옮김

#MAX_TEXT_TOKENS = 1000
#MAX_SUMM_TOKENS = 200

#MAX_SENT_TOKENS = 80 # 아래 결과 반영
#MAX_SENTS = 40       # 아래 결과 참조

#
# 학습데이터 총 320182 건 중 텍스트 내 문장길이의 분포
#
#   32개 이하 : 313855 개 98.02%
#   40개 이하 : 318660 개 99.52%
#   48개 이하 : 319828 개 99.89%
#   56개 이하 : 320067 개 99.96%
#   64개 이하 : 320104 개 99.98%
#
# 평균    : 15.23
# 표준편차: 18.58
# 최소    :  3
# 일사분위: 10
# 이사분위: 14
# 삼사분위: 19
# 최대    : 3089
#
# --------------------------------------------------------
# n_tokens in a sent upto 60: 4068213: 84.30853318086197 %
# n_tokens in a sent upto 64: 4392301: 91.02484422492952 %
# n_tokens in a sent upto 68: 4392303: 91.02488567238234 %
# n_tokens in a sent upto 72: 4648322: 96.33055338359390 %
# n_tokens in a sent upto 76: 4654613: 96.46092634642568 %
# n_tokens in a sent upto 80: 4825376: 99.99977203900951 %
# n_tokens in a sent upto 88: 4825385: 99.99995855254718 %
# --------------------------------------------------------
#


logger = logging.getLogger(__name__)


def get_data_from_ai_hub(config, mode='train', text_list=None, summ_list=None):
    fnm = whoami()

    text_dir = config.text_dir

    data_path_dict = {
            'train': [f'{text_dir}/1.Training/법률문서/train.jsonl'
                     ,f'{text_dir}/1.Training/사설잡지/train.jsonl'
                     ,f'{text_dir}/1.Training/신문기사/train.jsonl'
                     ]
           ,'test' : [f'{text_dir}/2.Validation/법률문서/dev.jsonl'
                     ,f'{text_dir}/2.Validation/사설잡지/dev.jsonl'
                     ,f'{text_dir}/2.Validation/신문기사/dev.jsonl'
                     ]
    }
    file_paths = data_path_dict.get(mode, None)

    if file_paths is None:
        warning = f'mode({fnm}) not valid. "train" or "test" must be specified as mode'
        raise ValueError(warnign)

    if text_list is None:
        text_list = list()
        pass

    if summ_list is None:
        summ_list = list()
        pass

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                article = json.loads(line)
                text = article.get('article_original', None)
                summ = article.get('abstractive', None)

                if not text or not summ:
                    logger.info(f'{i + 1}th line: not valid in {file_path}')
                    continue

                text_list.append(text)
                summ_list.append(summ)
                pass
            pass
        pass
    return text_list, summ_list




def main():
    pass


if __name__ == '__main__':
    main()
    pass
