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

from misc import whoami, run_scandir, run_scandir_re
from misc import init_logger, set_seed, load_config, get_gpu

from data import GNN_Dataset, TNS_Dataset, BatchSampler, GNN_Collator, TNS_Collator, create_masks
from model import get_model


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')

    parser.add_argument('--config-dir'    ,type=str , default=os.path.join('.' 'config'), help='config. directory')
    parser.add_argument('--config-file'   ,type=str , required=True, help='config. file in config. directory')
    parser.add_argument('--gpu_id'        ,type=int , default=0, help='GPU id')

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


def evaluate(config, model, dataloader, mode, global_step=None):
    results = dict()

    # start evaluation
    if global_step is not None:
        logger.info(f"***** Running evaluation on {mode} dataloader ({global_step} step) *****")
        pass
    else:
        logger.info(f"***** Running evaluation on {mode} dataloader *****")
        pass

    logger.info(f'  Num examples = {len(dataloader)}')
    logger.info(f'  Eval Batch size = {config.eval_batch_size}')

    device = config.device
    no_cuda = config.no_cuda

    eval_loss = 0.0
    eval_step = 0
    preds = None
    out_label_ids = None

    #for batch in progress_bar(dataloader):
    for batch in dataloader:
        model.eval()

        #batch = tuple(t.to(config.device) for t in batch) # YOYO

        # this where evaluation step goes
        with torch.no_grad():
            if config.tns_only:
                T, TN, V, S = batch
                pass
            else:
                T, TN, B, BN, V, S, A1, A2 = batch

                # text mask
                H = torch.cat([T, B], 1)
                pass

            # text mask
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

            ys = S[:, 1:].contiguous().view(-1)
            if not no_cuda:
                ys = ys.to(device)
                pass

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=config.pad_token_id)
            eval_loss += loss.item()
            pass

        eval_step += 1
        pass

    eval_loss = eval_loss / eval_step
    logger.info(f'eval_loss: {eval_loss:.4f}, eval_steps:{eval_step}')

    pass


def get_model_dir(config):
    fnm = whoami()

    now = datetime.datetime.now(pytz.timezone(config.timezone))
    now_str = f'{now.year:0>4}-{now.month:0>2}-{now.day:0>2}_{now.hour:0>2}:{now.minute:0>2}:{now.second:0>2}'
    return os.path.join(config.model_dir, now_str)


def train_model(config, model, train_dataset, test_dataset=None):
    fnm = whoami()

    collator = TNS_Collator(config) if config.tns_only else GNN_Collator(config)

    train_dataloader = DataLoader(train_dataset, \
                                  batch_size=config.train_batch_size, \
                                  num_workers=config.n_workers_for_dataloader, \
                                  collate_fn=collator, \
                                  sampler=BatchSampler(config.train_batch_size, len(train_dataset)))
    logger.info(f'{fnm}: train data loader generated')

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, \
                                 batch_size=config.eval_batch_size, \
                                 num_workers=config.n_workers_for_dataloader, \
                                 collate_fn=collator, \
                                 sampler=test_sampler )
    logger.info(f'{fnm}: test data loader generated')

    if 0 < config.max_steps:
        t_toal = config.max_steps
        config.num_train_epochs = config.max_steps // (len(train_dataloader) // config.gradient_accumulation_steps) + 1
        logger.info(f'{fnm}: t_total:{t_total}')
        logger.info(f'{fnm}: epochs:{config.num_train_epochs} = max_steps:{config.max_steps} // n_data:{len(train_dataloader)} // gradient_accumulation_steps:{config.gradient_accumulation_steps} + 1')
        pass
    else:
        t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
        logger.info(f'{fnm}: t_total:{t_total} = n_data:{len(train_dataloader)} // gradient_accumulation_steps:{config.gradient_accumulation_steps} * epochs:{config.num_train_epochs}')
        pass

    logger.info('#' * 80)

    n_params = len([model.named_parameters()])

    logger.info(f'{fnm}: n_naemd_params:{n_params}')

    logger.info('#' * 80)

    no_decay = ['bias', 'LayerNorm.weight']

    #
    # optimizer_grouped_parameters
    #
    optimizer_grouped_parameters = [
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params'      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0
        }
    ]
    logger.info('#' * 80)

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup( optimizer
                                               , num_warmup_steps=int(t_total * config.warmup_proportion)
                                               , num_training_steps=t_total
                                               )
    optimizer_path = os.path.join(config.model_name_or_path, 'optimizer.pt')
    scheduler_path = os.path.join(config.model_name_or_path, 'scheduler.pt')

    logger.info(f'optimizer_path:{optimizer_path}')
    logger.info(f'scheduler_path:{scheduler_path}')

    if os.path.isfile(optimizer_path) and os.path.isfile(scheduler_path):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path))
        scheduler.load_state_dict(torch.load(scheduler_path))
        pass

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Total train batch size = %d", config.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", config.logging_steps)
    logger.info("  Save steps = %d", config.save_steps)

    logger.info(f'model.training: {model.training}')
    logger.info(f'dir(model): {dir(model)}')
    logger.info(f'hasattr(model, "module"): {hasattr(model, "module")}')
    logger.info('#' * 80)
    logger.info(f'model:\n{model}')
    logger.info('#' * 80)

    device = config.device
    no_cuda = config.no_cuda

    model_dir = get_model_dir(config)
    logger.info(f'model_dir: [{model_dir}]')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        pass

    torch.save(config, os.path.join(model_dir, 'config.bin'))
    logger.info(f'config saved model to {model_dir}')

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()

    #for epoch in mb:
    for epoch in range(config.num_train_epochs):
        #epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(train_dataloader):
            model.train()

            # this where training step goes
            if config.tns_only:
                T, TN, V, S = batch
                pass
            else:
                T, TN, B, BN, V, S, A1, A2 = batch

                # text mask
                H = torch.cat([T, B], 1)
                pass

            # text mask
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

            ys = S[:, 1:].contiguous().view(-1)

            if not no_cuda:
                ys = ys.to(device)
                pass

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=config.pad_token_id)
            logger.info(f'{fnm}: epoch:{epoch:>2} / step:{step:>5} / global_step: {global_step:>5} / loss:{loss.item():.6f}')

            if 1 < config.gradient_accumulation_steps:
                loss = loss / config.gradient_accumulation_steps
                pass

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0 or \
               (len(train_dataloader) <= config.gradient_accumulation_steps \
                and (step + 1) == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if 0 < config.logging_steps and global_step % config.logging_steps == 0:
                    if config.evaluate_test_during_training:
                        evaluate(config, model, test_dataloader, 'test', global_step)
                        pass
                    pass

                if 0 < config.save_steps and global_step % config.save_steps == 0:
                    # Save model checkpoint
                    model_path = os.path.join(model_dir, f'model-{global_step:0>5}.pth')

                    torch.save(model.state_dict(), model_path)
                    logger.info(f'model saved to {model_path}')

                    if config.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(model_dir, 'scheduler.pt'))

                        logger.info(f'optimizer and scheduler states to {model_dir}')
                        pass
                    pass

                pass

            if 0 < config.max_steps and config.max_steps < global_step:
                logger.info(f'{fnm}: config.max_steps:{config.max_steps} / global_stpes:{global_step}, break...')
                break

            pass # end of for step, batch in ...
        pass # end of for epoch in ...

    # final evaluation
    evaluate(config, model, test_dataloader, 'test', global_step)

    # Save model checkpoint
    model_path = os.path.join(model_dir, f'model-{global_step:0>5}.pth')

    torch.save(model.state_dict(), model_path)
    logger.info(f'model saved to {model_path}')

    if config.save_optimizer:
        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(model_dir, 'scheduler.pt'))

        logger.info(f'optimizer and scheduler states to {model_dir}')
        pass

    return global_step, tr_loss
            

def get_dataset(config, mode):
    data_file_path = os.path.join(config.data_dir, f'ids.{mode}.pkl')
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)
        pass

    dataset = GNN_Dataset(config, mode, data)
    return dataset


def main():
    fnm = whoami()

    init_logger()

    args = get_args()
    logger.info(f'{fnm}: args:\n{args}')

    config_file_path = os.path.join(args.config_dir, args.config_file)
    config = load_config(config_file_path)

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

    train_data_file_path = os.path.join(config.data_dir, 'ids.train.pkl')
    with open(train_data_file_path, 'rb') as f:
        train_data = pickle.load(f)
        pass

    train_dataset = get_dataset(config, 'train')
    test_dataset  = get_dataset(config, 'test')

    if config.do_train:
        global_step, tr_loss = train_model(config, model, train_dataset, test_dataset)
        logger.info(f'{fnm}: global_step:{global_step}, average loss:{tr_loss}')
        pass

    pass


if __name__ == '__main__':
    main()
    pass
