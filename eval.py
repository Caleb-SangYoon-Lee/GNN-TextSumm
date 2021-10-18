# -*- coding: utf-8 -*-

import os, sys
import datetime
import glob
import re
import pandas as pd

import argparse
import logging
import pickle
import pytz

from transformers import  ElectraConfig, ElectraTokenizer, ElectraModel
from transformers import AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from ignite.metrics import Rouge

from misc import whoami, run_scandir, run_scandir_re
from misc import init_logger, set_seed, load_config, get_gpu

from data import GNN_Dataset, TNS_Dataset, BatchSampler, GNN_Collator, TNS_Collator, create_masks
from model import get_model


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')

    parser.add_argument('--data-dir'       ,type=str , required=True, help='root data directory')
    parser.add_argument('--model-dir'     ,type=str , required=True, help='model directory in which model image and config. are saved')
    parser.add_argument('--model-name'    ,type=str , default='', help='specific model file name in model_dir')
    parser.add_argument('--gpu-id'        ,type=int , default=0, help='GPU id')

    return parser.parse_args()


def get_dataset_from_dict_file(file_path, feature_names=None):
    fnm = whoami()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{fnm}: file({file_path}) not found')
        return

    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
        pass

    logger.info(f'{fnm}: data_dict loaded from {file_path}')

    arg_list = list()

    for feature_name in feature_names or data_dict:
        tensors = data_dict[feature_name]

        logger.info(f'{fnm}: {feature_name} - len: {len(tensors)} / size:{tensors.size()}')
        for i in range(3):
            logger.info(f'{fnm}: {i:>5}#: tensor len: {len(tensors[i])}:{tensors[i]}')
            pass

        arg_list.append(tensors)

        logger.info('#' * 80)
        pass

    dataset = TensorDataset(*arg_list)
    return dataset


def get_model_dir(config):
    fnm = whoami()

    now = datetime.datetime.now(pytz.timezone(config.timezone))
    now_str = f'{now.year:0>4}-{now.month:0>2}-{now.day:0>2}_{now.hour:0>2}:{now.minute:0>2}:{now.second:0>2}'
    return os.path.join(config.model_dir, now_str)


def eval_model(config, model, test_dataset):
    fnm = whoami()

    collator = TNS_Collator(config) if config.tns_only else GNN_Collator(config)

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, \
                                 batch_size=1, \
                                 num_workers=config.n_workers_for_dataloader, \
                                 collate_fn=collator, \
                                 sampler=test_sampler )
    logger.info(f'{fnm}: test data loader(n_data:{len(test_dataloader)}) generated')

    n_params = len([model.named_parameters()])

    logger.info(f'{fnm}: n_naemd_params:{n_params}')

    logger.info('#' * 80)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(test_dataset))

    logger.info(f'model.training: {model.training}')
    logger.info(f'dir(model): {dir(model)}')
    logger.info(f'hasattr(model, "module"): {hasattr(model, "module")}')
    logger.info('#' * 80)
    logger.info(f'model:\n{model}')
    logger.info('#' * 80)

    device = config.device
    no_cuda = config.no_cuda

    model.eval()

    pad_token_id = config.pad_token_id

    rouge_list = list()

    m = Rouge(variants=['L', 9], multiref='best', alpha=0.5) # alpah: 0.5 --> Rouge-L-F: F1 score
    n_data = len(test_dataset)

    for step, batch in enumerate(test_dataloader):
        # this where training step goes
        if config.tns_only:
            T, TN, V, S = batch
            pass
        else:
            T, TN, B, BN, V, S, A1, A2 = batch

            # text mask
            H = torch.cat([T, B], 1)
            pass

        S_ = S[:,:-1]

        if config.tns_only:
            HM, SM = create_masks(T, S_, config.pad_token_id)
            pass
        else:
            HM, SM = create_masks(H, S_, config.pad_token_id)
            pass

        if not no_cuda:
            T = T.to(device)
            HM = HM.to(device)
            S_ = S_.to(device)
            SM = SM.to(device)
            V = V.to(device)

            if not config.tns_only:
                B = B.to(device)
                A1 = A1.to(device)
                A2 = A2.to(device)
                pass
            pass

        if config.tns_only:
            preds = model(T, TN, HM, S_, SM, V)
            pass
        else:
            preds = model(T, TN, B, BN, HM, S_, SM, V, A1, A2)
            pass

        preds = torch.argmax(preds, dim=-1)
        preds = preds.squeeze()

        for i, id_ in enumerate(preds):
            if id_ == pad_token_id:
                preds = preds[:i]
                break
            pass

        preds = preds.cpu()

        pred_tokens = config.tokenizer.convert_ids_to_tokens(preds)
        pred_sent = config.tokenizer.convert_tokens_to_string(pred_tokens)

        ys = S[:, 1:].contiguous().view(-1)

        for i, id_ in enumerate(ys):
            if id_ == pad_token_id:
                ys = ys[:i]
                break
            pass

        ys_tokens = config.tokenizer.convert_ids_to_tokens(ys)
        ys_sent = config.tokenizer.convert_tokens_to_string(ys_tokens)

        m.update((pred_tokens, ys_tokens))

        rouge = m.compute()

        rouge_list.append(rouge)

        if (step + 1) % 100 == 0:
            n_curr = step + 1
            logger.info(f'{n_curr:>5} / {n_data:>5}: {n_curr * 100. / n_data:5.2f} %')
            pass

        pass # end of for step, batch in ...

    df = pd.DataFrame(rouge_list)

    return df
            

def get_dataset(config, mode):
    data_file_path = os.path.join(config.data_dir, f'ids.{mode}.pkl')
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)
        pass

    if config.tns_only:
        dataset = TNS_Dataset(config, mode, data)
        pass
    else:
        dataset = GNN_Dataset(config, mode, data)
        pass

    return dataset


def get_global_step_from_model_file_path(model_file_path:str) -> int:
    d, f = os.path.split(model_file_path)

    prefix, ext = os.path.splitext(f)
    global_step = prefix.split('-')[-1]

    return int(global_step)


def main():
    fnm = whoami()

    init_logger()

    args = get_args()
    logger.info(f'{fnm}: args:\n{args}')

    model_dir = os.path.join(args.data_dir, args.model_dir)

    config_file_path = os.path.join(model_dir, 'config.bin')
    if os.path.exists(config_file_path):
        logger.info(f'config. file({config_file_path}) found.')
        pass
    else:
        logger.info(f'config. file({config_file_path}) not found...')
        return

    with open(config_file_path, 'rb') as f:
        config = torch.load(f)
        pass

    logger.info(f'{fnm}: config:\n{config}')

    if not config.use_transformer_encoder and config.tns_only:
        logger.info(f'{fnm}: config.use_transformer_encoder:{config.use_transformer_encoder} and config.tns_only:{config.tns_only} conflicted..')
        return

    set_seed(config)
    logger.info(f'{fnm}: set_seed() done')
    logger.info('-' * 30)

    # set device
    if config.no_cuda:
        config.device = 'cpu'
        pass
    else:
        config.device = get_gpu(gpu_id=args.gpu_id, print_gpu_info=True)
        pass

    logger.info(f'{fnm}: config.no_cuda:{config.no_cuda} / config.device:{config.device}')


    # --------------
    # init. tokenizer
    # --------------
    nlp_model_type = config.model_type
    nlp_model_path = config.model_name_or_path
    do_lower_case  = config.do_lower_case

    logger.info(f'{fnm}: nlp_model_type:{nlp_model_type} / nlp_model_path:{nlp_model_path} / do_lower_case:{do_lower_case}')

    tokenizer = ElectraTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    logger.info(f'tokenizer({nlp_model_type}) with {config.model_name_or_path} / {config.do_lower_case} done')

    lm = ElectraModel.from_pretrained(config.model_name_or_path)  # language model for KoELECTRA-Base-v3
    logger.info(f'{fnm}: language model({config.model_name_or_path}) loaded')

    if not config.no_cuda:
        lm = lm.to(config.device)
        pass

    logger.info(f'{fnm}:#1: lm.embeddings.word_embeddings.training:{lm.embeddings.word_embeddings.training}')

    config.tokenizer = tokenizer
    config.lm = lm

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id

    logger.info(f'{fnm}: config.cls_token_id: {config.cls_token_id}')
    logger.info(f'{fnm}: config.sep_token_id: {config.sep_token_id}')
    logger.info(f'{fnm}: config.pad_token_id: {config.pad_token_id}')

    # --------------

    model = get_model(config)
    logger.info('get_model() done..')

    if 'model_name' in args  and  args.model_name:
        model_path = os.path.join(model_dir, args.model_name)
        logger.info(f'model_path: {model_path}')
        pass
    else:
        model_file_names = glob.glob(os.path.join(model_dir, '*.pth'))
        model_file_names.sort(key=get_global_step_from_model_file_path)
        model_path = model_file_names[-1]

        logger.info(f'args.model_name not specified, [{model_path}] used...')
        pass

    model.load_state_dict(torch.load(model_path))
    global_step = get_global_step_from_model_file_path(model_path)

    logger.info(f'model state loaded from {model_path}, global_step:{global_step}')

    test_dataset = get_dataset(config, 'test')
    logger.info(f'length of test dataset: {len(test_dataset)}')

    if config.do_eval:
        df = eval_model(config, model, test_dataset)

        eval_result_file_name = 'eval-df.bin'
        eval_result_file_path = os.path.join(model_dir, eval_result_file_name)

        torch.save(df, eval_result_file_path)
        logger.info(f'evaluation result saved in [{eval_result_file_path}]')

        logger.info(df.describe())
        pass

    pass


if __name__ == '__main__':
    main()
    pass
