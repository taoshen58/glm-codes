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
from utils.mutli_proc import combine_from_lists


class LinkPredictionDataset(KbDataset):
    def __init__(self, *arg, **kwargs):
        super(LinkPredictionDataset, self).__init__(*arg, **kwargs)

    def __getitem__(self, item):
        raw_data, label = super(LinkPredictionDataset, self).__getitem__(item)
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
    def batch2feed_dict(cls, batch, data_format=None):
        inputs = {
            'input_ids': batch[0],  # bs, sl
            'attention_mask': batch[1],  #
            'token_type_ids': batch[2],  #
            "labels": batch[-1],  #
        }
        return inputs


class LinkPredictionInferReuseDataset(Dataset):
    def __init__(self, eval_dataset, ent_list, rel_list):
        self.eval_dataset = eval_dataset

        # ent2tkids and rel2tkids
        ent2tkids, rel2tkids = {}, {}
        for ent in ent_list:
            ent_ids = self.eval_dataset.str2ids(self.eval_dataset.ent2text[ent])[1:-1]
            ent2tkids[ent] = ent_ids
        for rel in rel_list:
            rel_ids = self.eval_dataset.str2ids(self.eval_dataset.rel2text[rel])[1:-1]
            rel2tkids[rel] = rel_ids
        self.ent2tkids, self.rel2tkids = ent2tkids, rel2tkids

        self._cls_id, self._sep_id = self.eval_dataset._cls_id, self.eval_dataset._sep_id

        # these two will be updated during predicting
        self.triplet_list = None
        self.triplet_ids_list = None
        self.original_permutation = None  # for sort dev

    def update_triplet_list(self, new_triplet_list): # return the original permutation
        # self.triplet_list = triplet_list

        tmp_triplet_ids_list = []
        len_list = []
        for triplet in new_triplet_list:
            head, rel, tail = triplet[:3]
            head_ids = self.ent2tkids[head]
            rel_ids = self.rel2tkids[rel]
            tail_ids = self.ent2tkids[tail]
            tmp_triplet_ids_list.append([head_ids, rel_ids, tail_ids])
            len_list.append(len(head_ids) + len(rel_ids) + len(tail_ids))

        # sort
        permutation = list(np.argsort(len_list))
        triplet_list = [new_triplet_list[_idx_ex] for _idx_ex in permutation]
        triplet_ids_list = [tmp_triplet_ids_list[_idx_ex] for _idx_ex in permutation]

        self.triplet_list = triplet_list
        self.triplet_ids_list = triplet_ids_list
        self.original_permutation = permutation
        return permutation

    def recover_original_order(self, new_order_array):
        org_order_array = np.zeros_like(new_order_array)
        for _new_idx, _org_idx in enumerate(self.original_permutation):
            org_order_array[_org_idx] = new_order_array[_new_idx]
        return org_order_array

    def __len__(self):
        return len(self.triplet_ids_list)

    def __getitem__(self, item):
        assert self.triplet_list is not None

        head_ids, rel_ids, tail_ids = self.triplet_ids_list[item][:3]

        remain_len = self.eval_dataset.max_seq_length - 4 - len(rel_ids)
        assert remain_len >= 4
        while len(head_ids) + len(tail_ids) > remain_len:
            if len(head_ids) > len(tail_ids):
                head_ids.pop(-1)
            else:
                tail_ids.pop(-1)
        input_ids = [self._cls_id] + head_ids + [self._sep_id] + rel_ids + [self._sep_id] + tail_ids + [self._sep_id]
        type_ids = [0] * (len(head_ids) + 2) + [1] * (len(rel_ids) + 1) + [0] * (len(tail_ids) + 1)
        mask_ids = [1] * len(input_ids)
        return_list = [
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(mask_ids, dtype=torch.long),
            torch.tensor(type_ids, dtype=torch.long),
        ]
        return tuple(return_list)

    def data_collate_fn(self, batch):
        return self.eval_dataset.data_collate_fn(batch)

    @classmethod
    def batch2feed_dict(cls, batch, data_format=None):
        inputs = {
            'input_ids': batch[0],  # bs, sl
            'attention_mask': batch[1],  #
            'token_type_ids': batch[2],  #
        }
        return inputs


def predict(args, raw_examples, dataset_list, model, verbose=True):
    logging.info("***** Running Prediction*****")
    model.eval()
    # get the last one (i.e., test) to make use if its useful functions and data
    standard_dataset = dataset_list[-1]

    ents = set()
    g_subj2objs = collections.defaultdict(lambda: collections.defaultdict(set))
    g_obj2subjs = collections.defaultdict(lambda: collections.defaultdict(set))
    for _ds in dataset_list:
        for _raw_ex in _ds.raw_examples:
            _head, _rel, _tail = _raw_ex
            ents.add(_head)
            ents.add(_tail)
            g_subj2objs[_head][_rel].add(_tail)
            g_obj2subjs[_tail][_rel].add(_head)
    ent_list = list(sorted(ents))
    rel_list = standard_dataset.rel_list
    # rel2idx = eval_dataset.rel2idx

    # ========= run link prediction ==========

    # * begin to get hit
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    top_ten_hit_count = 0
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    infer_reuse_dataset = LinkPredictionInferReuseDataset(standard_dataset, ent_list, rel_list)

    for _idx_ex, _triplet in enumerate(tqdm(raw_examples, desc="evaluating")):
        _head, _rel, _tail = _triplet

        head_ent_list = []
        tail_ent_list = []

        # head corrupt
        _pos_head_ents = g_obj2subjs[_tail][_rel]
        _neg_head_ents = ents - _pos_head_ents
        head_ent_list.append(_head)  # positive example
        head_ent_list.extend(_neg_head_ents)  # negative examples
        tail_ent_list.extend([_tail] * (1 + len(_neg_head_ents)))
        split_idx = len(head_ent_list)

        # tail corrupt
        _pos_tail_ents = g_subj2objs[_head][_rel]
        _neg_tail_ents = ents - _pos_tail_ents
        head_ent_list.extend([_head] * (1 + len(_neg_tail_ents)))
        tail_ent_list.append(_tail)  # positive example
        tail_ent_list.extend(_neg_tail_ents)  # negative examples

        triplet_list = list([_h, _rel, _t] for _h, _t in zip(head_ent_list, tail_ent_list))

        # infer_dataset = LinkPredictionInferDataset(eval_dataset, triplet_list)
        infer_reuse_dataset.update_triplet_list(triplet_list)
        infer_dataloader = setup_eval_step(
            args, infer_reuse_dataset, collate_fn=infer_reuse_dataset.data_collate_fn,
            num_workers=args.num_workers * 2)

        local_scores_list = []
        model.eval()
        for batch in tqdm(infer_dataloader, desc="Evaluating", disable=(not verbose)):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = LinkPredictionInferReuseDataset.batch2feed_dict(batch)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs[0]
                if True:  # todo: open this for probs
                    logits = torch.softmax(logits, dim=-1)
                local_scores = logits.detach().cpu().numpy()[:, 1]
            local_scores_list.append(local_scores)
        scores = np.concatenate(local_scores_list, axis=0)
        # permutation
        scores = infer_reuse_dataset.recover_original_order(scores)

        # left
        left_scores = scores[:split_idx]
        left_sort_idxs = np.argsort(-left_scores)
        left_rank = np.where(left_sort_idxs == 0)[0][0]
        ranks_left.append(left_rank + 1)
        ranks.append(left_rank + 1)
        # print("left_scores", left_scores[:20])
        # print("left_sort_idxs", left_sort_idxs[:20])
        # print("left_rank", left_rank)


        # right
        right_sscores = scores[split_idx:]
        right_sort_idxs = np.argsort(-right_sscores)
        right_rank = np.where(right_sort_idxs == 0)[0][0]
        ranks_right.append(right_rank + 1)
        ranks.append(right_rank + 1)
        # print("right_sscores", right_sscores[:20])
        # print("right_sort_idxs", right_sort_idxs[:20])
        # print("right_rank", right_rank)

        # log
        top_ten_hit_count += (int(left_rank < 10) + int(right_rank < 10))
        if (_idx_ex + 1) % 10 == 0:
            logger.info("hit@10 until now: {}".format(top_ten_hit_count * 1.0 / len(ranks)))
            logger.info('mean rank until now: {}'.format(np.mean(ranks)))

            # print("left: {}/{}; right: {}/{}".format(
            #     left_rank, split_idx, right_rank, len(head_ent_list) - split_idx)
            # )

        # hits
        for hits_level in range(10):
            if left_rank <= hits_level:
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if right_rank <= hits_level:
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)
    if verbose:
        for i in [0, 2, 9]:
            logger.info('Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])))
            logger.info('Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])))
            logger.info('Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])))
        logger.info('Mean rank left: {0}'.format(np.mean(ranks_left)))
        logger.info('Mean rank right: {0}'.format(np.mean(ranks_right)))
        logger.info('Mean rank: {0}'.format(np.mean(ranks)))
        logger.info('Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))))
        logger.info('Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))))
        logger.info('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

    tuple_ranks = [[int(_l), int(_r)] for _l, _r in zip(ranks_left, ranks_right)]
    return tuple_ranks


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
        inputs = LinkPredictionDataset.batch2feed_dict(batch)
        with torch.no_grad():
            outputs = model(**inputs)  #
            tmp_eval_loss, logits = outputs[:2]
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()  # bs,2
            tmp_eval_loss = tmp_eval_loss.mean().item()
            labels = inputs["labels"].cpu().numpy()  # bs
        eval_loss += tmp_eval_loss
        # accuracy
        tmp_eval_accuracy = np.sum((np.argmax(probs, axis=-1)).astype(labels.dtype) == labels)
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
            inputs = LinkPredictionDataset.batch2feed_dict(batch)

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
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", default="roberta", type=str,
                        help="model class, one of [bert, roberta]")
    parser.add_argument("--dataset", type=str, default="wn18rr")
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--neg_weights", default=None, type=str)

    # extra parameters for prediction
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--collect_prediction", action="store_true")
    parser.add_argument("--prediction_part", default="0,1", type=str)


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
        model_class = RobertaForSequenceClassification
    elif args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        model_class = BertForSequenceClassification
    else:
        raise KeyError(args.model_class)

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, num_labels=2)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Dataset
    neg_weights = [1., 1., 0.] if args.neg_weights is None else [float(_e) for _e in args.neg_weights.split(",")]
    assert len(neg_weights) == 3 and sum(neg_weights) > 0

    train_dataset = LinkPredictionDataset(
        args.dataset, "train", None, data_dir,
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length, neg_times=5, neg_weights=neg_weights
    )
    dev_dataset = LinkPredictionDataset(
        args.dataset, "dev", None, data_dir,
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )
    test_dataset = LinkPredictionDataset(
        args.dataset, "test", None, data_dir,
        args.model_class, tokenizer, args.do_lower_case,
        args.max_seq_length,
    )

    if args.do_train:
        train(args, train_dataset, model, tokenizer, eval_dataset=dev_dataset)

    if args.do_train and (args.do_eval or args.do_prediction):  # load the best model
        model = model_class.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

    if not args.do_train and args.do_eval:
        pass

    if args.fp16:
        model = setup_eval_model_for_fp16(args, model)

    dataset_list = [train_dataset, dev_dataset, test_dataset]

    if not args.do_train and args.do_prediction:
        path_template = join(args.output_dir, "tuple_ranks_{},{}.json")
        part_param = args.prediction_part.split(",")
        part_param = [int(_e) for _e in part_param]
        assert len(part_param) == 2 and part_param[1] > part_param[0] >= 0
        cur_part_idx, num_parts = part_param

        if args.collect_prediction:
            tuple_ranks_list = []
            for _idx in range(num_parts):
                tuple_ranks_list.append(load_json(path_template.format(_idx, num_parts)))
            tuple_ranks = combine_from_lists(tuple_ranks_list, ordered=True)
            output_str = calculate_metrics_for_link_prediction(tuple_ranks)
            with open(join(args.output_dir, "link_prediction_metrics.txt"), "w", encoding="utf-8") as fp:
                fp.write(output_str)
        else:
            test_raw_examples = test_dataset.raw_examples
            # part
            tgt_raw_examples = [_ex for _idx, _ex in enumerate(test_raw_examples) if _idx%num_parts == cur_part_idx]
            # evaluate(args, test_dataset, model, tokenizer, None, "test_")
            tuple_ranks = predict(
                args, tgt_raw_examples, dataset_list, model, verbose=(not args.no_verbose))
            calculate_metrics_for_link_prediction(tuple_ranks, verbose=True)
            save_json(tuple_ranks, path_template.format(cur_part_idx, num_parts))


def calculate_metrics_for_link_prediction(tuple_ranks, verbose=True):
    ranks_left, ranks_right, ranks = [], [], []
    hits_left, hits_right, hits = [], [], []
    for i in range(10):
        hits_left.append([])
        hits_right.append([])
        hits.append([])

    for _left_rank, _right_rank in tuple_ranks:
        ranks.append(_left_rank)
        ranks.append(_right_rank)
        ranks_left.append(_left_rank)
        ranks_right.append(_right_rank)

        # hits
        for hits_level in range(10):
            if _left_rank <= hits_level+1:  # because numbers in tuple_ranks start with 1
                hits[hits_level].append(1.0)
                hits_left[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_left[hits_level].append(0.0)

            if _right_rank <= hits_level+1:  # because numbers in tuple_ranks start with 1
                hits[hits_level].append(1.0)
                hits_right[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)
                hits_right[hits_level].append(0.0)

    output_str = ""
    linesep = os.linesep

    for i in [0, 2, 9]:
        output_str += 'Hits left @{0}: {1}'.format(i + 1, np.mean(hits_left[i])) + linesep
        output_str += 'Hits right @{0}: {1}'.format(i + 1, np.mean(hits_right[i])) + linesep
        output_str += 'Hits @{0}: {1}'.format(i + 1, np.mean(hits[i])) + linesep
    output_str += 'Mean rank left: {0}'.format(np.mean(ranks_left)) + linesep
    output_str += 'Mean rank right: {0}'.format(np.mean(ranks_right)) + linesep
    output_str += 'Mean rank: {0}'.format(np.mean(ranks)) + linesep
    output_str += 'Mean reciprocal rank left: {0}'.format(np.mean(1. / np.array(ranks_left))) + linesep
    output_str += 'Mean reciprocal rank right: {0}'.format(np.mean(1. / np.array(ranks_right))) + linesep
    output_str += 'Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))) + linesep

    if verbose:
        logger.info(output_str)
    return output_str


if __name__ == '__main__':
    main()