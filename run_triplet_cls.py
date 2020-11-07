from configs import USER_HOME, kgbert_data_dir
from nn_utils.help import *
from torch.utils.data import Dataset, TensorDataset, DataLoader
from os.path import join
from utils.common import save_json, load_json, save_list_to_file, load_list_from_file, file_exists
from tqdm import tqdm, trange
import collections
import logging as logger
import numpy as np
import random
import torch
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification
from tensorboardX import SummaryWriter
import argparse
from kbc.kb_dataset import KbDataset


class TripletClsDataset(KbDataset):
    def __init__(self, *arg, **kwargs):
        super(TripletClsDataset, self).__init__(*arg, **kwargs)
        self.negative_samples = []
        for _ex in self.raw_examples:
            self.negative_samples.append(self.negative_sampling(_ex, weights=self.neg_weights))


    def __getitem__(self, item):
        raw_data, label = super(TripletClsDataset, self).__getitem__(item)

        assert self.data_format == "cat"

        features = self.convert_raw_example_to_features(raw_data, method="2")
        input_ids, type_ids = features
        mask_ids = [1] * len(input_ids)

        return_list = [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(mask_ids, dtype=torch.long),
            torch.tensor(type_ids, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        ]
        return tuple(return_list)

    @classmethod
    def batch2feed_dict(cls, batch, data_format):
        if data_format == "cat":
            inputs = {
                'input_ids': batch[0],  # bs, sl
                'attention_mask': batch[1],  #
                'token_type_ids': batch[2],  #
                "labels": batch[-1],  #
            }
        else:
            raise KeyError(data_format)
        return inputs


def evaluate(args, eval_dataset, model, tokenizer, global_step=None, file_prefix=""):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)

    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn, num_workers=args.num_workers * 2)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    _idx_ex = 0
    _eval_predict_data = []
    preds_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = TripletClsDataset.batch2feed_dict(batch, args.data_format)

        with torch.no_grad():
            outputs = model(**inputs)  #
            tmp_eval_loss, logits = outputs[:2]

            labels = inputs["labels"].to('cpu').numpy()
            logits = logits.detach().cpu().numpy()

        eval_loss += tmp_eval_loss.mean().item()
        # accuracy
        tmp_eval_accuracy = np.sum(np.argmax(logits, axis=-1) == labels)
        eval_accuracy += tmp_eval_accuracy

        # count
        nb_eval_examples += inputs["input_ids"].size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              # 'global_step': global_step,
              # 'loss': tr_loss / (global_step if global_step > 0 else 1)
              }

    output_eval_file = os.path.join(args.output_dir, file_prefix + "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logging.info("***** Eval results at {}*****".format(global_step))
        writer.write("***** Eval results at {}*****\n".format(global_step))
        for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n")
    return eval_accuracy


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
            inputs = TripletClsDataset.batch2feed_dict(batch, args.data_format)

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

                if args.local_rank in [-1, 0] and eval_dataset is not None \
                        and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    metric_cur = evaluate(
                        args, eval_dataset, model, tokenizer, global_step=global_step, file_prefix="eval_")
                    if metric_cur > metric_best:
                        save_model_with_default_name(args.output_dir, model, tokenizer, args)
                        metric_best = metric_cur
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        # can add epoch evaluation
        if args.local_rank in [-1, 0] and eval_dataset is not None and args.eval_steps <= 0:
            metric_cur = evaluate(
                args, eval_dataset, model, tokenizer, global_step=global_step, file_prefix="eval_")
            if metric_cur > metric_best:
                save_model_with_default_name(args.output_dir, model, tokenizer, args)
                metric_best = metric_cur

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    # final evaluation
    metric_cur = evaluate(
        args, eval_dataset, model, tokenizer, global_step=global_step, file_prefix="eval_")
    if metric_cur > metric_best:
        save_model_with_default_name(args.output_dir, model, tokenizer, args)
        metric_best = metric_cur

    with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
        fp.write("{}{}".format(metric_best, os.linesep))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str,
                        help="model class, one of [bert, roberta]")
    parser.add_argument("--dataset", type=str, default="wn18rr")
    parser.add_argument("--model_type", default=None, type=str,
                        help="model_type")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--neg_weights", default=None, type=str)
    parser.add_argument("--data_format", default="cat", type=str, help="cat|rel")
    ## Other parameters
    define_hparams_training(parser)
    args = parser.parse_args()

    data_dir = args.data_dir or kgbert_data_dir

    # setup
    setup_prerequisite(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        if args.data_format == "cat":
            model_class = RobertaForSequenceClassification
        else:
            raise KeyError(args.data_format)
    elif args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        if args.data_format == "cat":
            model_class = BertForSequenceClassification
        else:
            raise KeyError(args.data_format)
    else:
        raise NotImplementedError(args.model_class)

    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    config.num_labels = 2
    # build model
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    neg_weights  =[1., 1., 0.] if args.neg_weights is None else [float(_e) for _e in args.neg_weights.split(",")]
    assert len(neg_weights) == 3 and sum(neg_weights) > 0

    dataset_class = TripletClsDataset

    train_dataset = dataset_class(
        args.dataset, "train", args.data_format, data_dir,
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length, neg_times=1, neg_weights=neg_weights
    )

    dev_dataset = dataset_class(
        args.dataset, "dev", args.data_format, data_dir,
        args.model_class, tokenizer, args.do_lower_case, args.max_seq_length,
    )

    test_dataset = dataset_class(
        args.dataset, "test", args.data_format, data_dir,
        args.model_class, tokenizer, args.do_lower_case, args.max_seq_length,
    )

    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=dev_dataset)

    if args.do_train and args.do_eval:  # load the best model
        model = model_class.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

    if args.do_eval:
        dev_accu = evaluate(args, dev_dataset, model, tokenizer, None, "dev_final_")
        test_accu = evaluate(args, test_dataset, model, tokenizer, None, "test_final_")
        with open(os.path.join(args.output_dir, "predicted_results.txt"), "w") as fp:
            fp.write("{},{}{}".format(dev_accu, test_accu, os.linesep))


if __name__ == '__main__':
    main()
