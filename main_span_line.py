import argparse
import csv
import logging
import os
import random
import sys
import math
from io import open
import collections

import numpy as np
import torch
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from utils.common import get_val_str_from_dict

from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
# roberta
# from transformers import RobertaForMaskedLM
from nn_utils.roberta.glm import RobertaForGLM
# bert
from nn_utils.bert.baselines import BertForOurMaskedLM
from nn_utils.bert.span_bert import SpanBertForPreTraining
from nn_utils.bert.glm import BertForGLM

# from src.roberta.gnn import RobertaForCGLM, RobertaForCLM, RobertaForFCLM
# from src.bert.graph_bert import GraphBertForPreTraining, GraphBertForPreTrainingV1
# from src.bert.baselines import BertForPreTrainingMLM

from src_lm.data.dataset_line import DatasetLine
from configs import USER_HOME, index_sent_cache_dir

from nn_utils.help import *
import logging as logger


def evaluate(args, eval_dataset, model, tokenizer, global_step=None, file_prefix=""):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn, num_workers=args.num_workers*4)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    _idx_ex = 0
    _eval_predict_data = []
    preds_list = []

    all_res_loss_dict = collections.defaultdict(lambda: 0.)
    all_res_num_dict = collections.defaultdict(lambda: 0.)
    # all_num_masked = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = [_t.to(args.device) for _t in batch]
        inputs = DatasetLine.batch2feed_dict(batch, eval_dataset.data_format)

        with torch.no_grad():
            outputs = model(**inputs)

        res_dict = outputs[-1]
        # num_masked_np = res_dict.pop("num_masked").detach().cpu().numpy()
        # num_masked = num_masked_np.sum()
        # all_num_masked += num_masked

        for _k, _v in res_dict.items():
            if _k.startswith("loss"):
                _v_np = _v.detach().cpu().numpy()
                _v_np = _v_np[_v_np > 0.00001]
                all_res_loss_dict[_k] += _v_np.sum()
                all_res_num_dict[_k] += len(_v_np)
    for _k in all_res_loss_dict:
        all_res_loss_dict[_k] = all_res_loss_dict[_k] / all_res_num_dict[_k]
    res_str = get_val_str_from_dict(all_res_loss_dict)
    res_str = "At step {}, {}".format(global_step, res_str)

    logger.info(res_str)

    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        writer.write(res_str)
        writer.write(os.linesep)

    if "loss" in all_res_loss_dict:
        return - all_res_loss_dict["loss"]
    else:
        return 0.


def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # learning setup
    train_dataloader = setup_training_step(
        args, train_dataset, collate_fn=train_dataset.data_collate_fn, num_workers=args.num_workers)

    # Prepare optimizer and schedule (linear warmup and decay)
    model, optimizer, scheduler = setup_opt(args, model)
    metric_best = -1e5
    global_step = 0
    if args.local_rank in [-1, 0] and args.eval_at_beginning:
        evaluate(args, eval_dataset, model, tokenizer, global_step, file_prefix="dev_")
    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = DatasetLine.batch2feed_dict(batch, train_dataset.data_format)

            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss = update_wrt_loss(args, model, optimizer, loss)

            step_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                model_update_wrt_gradient(args, model, optimizer, scheduler)

                global_step += 1

                # update loss for logging
                ma_dict({"loss": step_loss})
                step_loss = 0.

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging.info(ma_dict.get_val_str())

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model_with_default_name(args.output_dir, model, tokenizer, args)

                if args.local_rank in [-1,
                                       0] and eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    metric_cur = evaluate(args, eval_dataset, model, tokenizer, global_step, file_prefix="dev_")
                    if metric_cur > metric_best:
                        save_model_with_default_name(args.output_dir, model, tokenizer, args)
                        metric_best = metric_cur
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and eval_dataset is not None and args.eval_steps <= 0:
            metric_cur = evaluate(args, eval_dataset, model, tokenizer, global_step, file_prefix="dev_")
            if metric_cur > metric_best:
                save_model_with_default_name(args.output_dir, model, tokenizer, args)
                metric_best = metric_cur

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str,
                        help="model class, one of [bert, roberta]")
    parser.add_argument("--data_type_list", type=str, default="omcs,arc")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="model_type")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--mask_proportion", default=0.20, type=float)
    parser.add_argument("--preproc_dir", default="", type=str,
                        help="")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--sent_format", default="cat", type=str, help="cat|single")
    parser.add_argument("--data_format", default="cn", type=str, help="[cn|wwm,span]")

    # add lambda
    parser.add_argument("--loss_lambda", default=0.2, type=float)
    parser.add_argument("--disable_rev_pos", action="store_true")

    parser.add_argument("--stop_proportion", default=0, type=int, help="Percentile ratio of stop token, 0 for dynamic")

    parser.add_argument("--use_simple_neg", action="store_true")
    parser.add_argument("--use_invalid_omcs", action="store_true")
    parser.add_argument("--use_nongraph", action="store_true")

    parser.add_argument("--only_dataset", action="store_true")

    ## Other parameters
    define_hparams_training(parser)
    args = parser.parse_args()

    # Parameter fix if needed
    args.use_nongraph = False
    if args.sent_format == "cat":
        args.max_seq_length = 240
        args.mask_proportion = 0.15
    elif args.sent_format == "single":
        args.max_seq_length = 80
        args.mask_proportion = 0.2
    else:
        assert KeyError(args.sent_format)
    if args.model_class == "roberta":
        args.adam_betas = "0.9,0.98"
        args.adam_epsilon = 1e-6
        args.max_grad_norm = 0.
        args.weight_decay = 0.01
        args.warmup_proportion = 0.05
        args.logging_steps = 100
    elif args.model_class == "bert":
        args.warmup_proportion = 0.1
        args.logging_steps = 100

    # setup
    setup_prerequisite(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        if args.model_type == "glm":
            model_class = RobertaForGLM
        else:
            raise NotImplementedError(args.model_type)
    elif args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        if args.model_type == "mlm":
            model_class = BertForOurMaskedLM
        elif args.model_type == "span":
            model_class = SpanBertForPreTraining
        elif args.model_type == "glm":
            model_class = BertForGLM
        else:
            raise NotImplementedError(args.model_type)
    else:
        raise NotImplementedError(args.model_class)

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    # add more args
    config.loss_lambda = args.loss_lambda
    config.disable_rev_pos = args.disable_rev_pos
    # build model
    model = None
    if not args.only_dataset:
        model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if model is not None:
        model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    train_dataset, dev_dataset = DatasetLine.build_training_and_dev_datasets(
        args.data_type_list.split(","), args.do_lower_case, tokenizer, args.max_seq_length,
        cached_dir=args.preproc_dir or index_sent_cache_dir,
        threshold_stop_ctk=args.stop_proportion, num_parallels=args.num_workers, mask_proportion=0.2,
        use_simple_neg=args.use_simple_neg, use_invalid_omcs=args.use_invalid_omcs, use_nongraph=args.use_nongraph,
        tokenizer_type=args.model_class, sent_format=args.sent_format, data_format=args.data_format,
        dev_proportion=5000,
    )
    if args.only_dataset:
        return

    if args.do_train:
        train(args, train_dataset, model, tokenizer, dev_dataset)


if __name__ == '__main__':
    main()
