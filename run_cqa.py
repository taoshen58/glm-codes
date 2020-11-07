

import argparse
import glob
import logging
import numpy as np
from io import open
import torch

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, Dataset

from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers import RobertaForMultipleChoice, BertForMultipleChoice

from utils.common import load_jsonl

from nn_utils.help import *


def accuracy_np(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


class CommonsenseqaDataset(Dataset):
    LABELS = ['A', 'B', 'C', 'D', 'E']
    DATA_TYPE_TO_FILENAME = {
        "train": 'train_{split}_split.jsonl'.format(split="rand"),
        "dev": 'dev_{split}_split.jsonl'.format(split="rand"),
        "test": 'test_{split}_split_no_answers.jsonl'.format(split="rand"),
    }

    @staticmethod
    def get_labels():
        return [0, 1, 2, 3, 4]

    def __init__(self, data_type, data_dir, tokenizer, do_lower_case, max_seq_length, **kwargs):
        self.data_type = data_type
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.do_lower_case = do_lower_case
        self.max_seq_length = max_seq_length

        self.data_path = os.path.join(self.data_dir, self.DATA_TYPE_TO_FILENAME[self.data_type])

        raw_examples = load_jsonl(self.data_path)

        self.example_list = []
        for line in raw_examples:
            qid = line['id']
            question = "Q: " + line['question']['stem']  # tokenization.convert_to_unicode(line['question']['stem'])
            answers = np.array([
                "A: " + choice['text']  # tokenization.convert_to_unicode(choice['text'])
                for choice in sorted(
                    line['question']['choices'],
                    key=lambda c: c['label'])
            ])
            # the test set has no answer key so use 'A' as a dummy label
            label = self.LABELS.index(line.get('answerKey', 'A'))
            example = {
                "qid": qid,
                "question": question,
                "answers": answers,
                "label": label
            }
            self.example_list.append(example)

        self.cls_token, self.sep_token, self.pad_token = \
            self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token
        self.cls_id, self.sep_id, self.pad_id = self.tokenizer.convert_tokens_to_ids(
            [self.cls_token, self.sep_token, self.pad_token])

    def get_all_qid(self):
        return [_ex["qid"] for _ex in self.example_list]

    def __getitem__(self, item):
        example = self.example_list[item]

        max_seq_length = self.max_seq_length
        question_tokens = self.tokenizer.tokenize(example["question"])
        answers_tokens = map(self.tokenizer.tokenize, example["answers"])

        tokens = []
        token_ids = []
        segment_ids = []
        for choice_idx, answer_tokens in enumerate(answers_tokens):
            truncated_question_tokens = question_tokens[
                                        :max((max_seq_length - 3) // 3 * 2, max_seq_length - (len(answer_tokens) + 3))]
            truncated_answer_tokens = answer_tokens[
                                      :max((max_seq_length - 3) // 3 * 1, max_seq_length - (len(question_tokens) + 3))]

            choice_tokens = []
            choice_segment_ids = []
            choice_tokens.append(self.cls_token)
            choice_segment_ids.append(0)
            for question_token in truncated_question_tokens:
                choice_tokens.append(question_token)
                choice_segment_ids.append(0)
            choice_tokens.append(self.sep_token)
            choice_segment_ids.append(0)
            for answer_token in truncated_answer_tokens:
                choice_tokens.append(answer_token)
                choice_segment_ids.append(1)
            choice_tokens.append(self.sep_token)
            choice_segment_ids.append(1)

            choice_token_ids = self.tokenizer.convert_tokens_to_ids(choice_tokens)

            tokens.append(choice_tokens)
            token_ids.append(choice_token_ids)
            segment_ids.append(choice_segment_ids)

        # padding
        cur_max_len = max(len(_e) for _e in tokens)
        choices_features = []

        for _idx_choice in range(len(tokens)):
            choice_tokens = tokens[_idx_choice]
            choice_token_ids = token_ids[_idx_choice]
            choice_segment_ids = segment_ids[_idx_choice]
            assert len(choice_tokens) <= max_seq_length, "{}/{}".format(len(choice_tokens), max_seq_length)
            assert len(choice_tokens) == len(choice_token_ids) == len(choice_segment_ids)

            padding_len = cur_max_len - len(choice_token_ids)
            padded_choice_token_ids = choice_token_ids + [self.pad_id] * padding_len
            padded_choice_token_mask = [1] * len(choice_token_ids) + [0] * padding_len
            padded_choice_segment_ids = choice_segment_ids + [0] * padding_len

            choices_features.append((choice_tokens, padded_choice_token_ids,
                                     padded_choice_token_mask, padded_choice_segment_ids))

        input_ids = torch.tensor([_e[1] for _e in choices_features], dtype=torch.long)
        mask_ids = torch.tensor([_e[2] for _e in choices_features], dtype=torch.long)
        segment_ids = torch.tensor([_e[3] for _e in choices_features], dtype=torch.long)
        label = torch.tensor(example["label"], dtype=torch.long)

        return input_ids, mask_ids, segment_ids, label

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))
        return_list = []
        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t == 0:
                padding_value = self.pad_id
            else:
                padding_value = 0

            if _tensors[0].dim() >= 1:
                _tensors = [_t.t() for _t in _tensors]
                return_list.append(
                    torch.nn.utils.rnn.pad_sequence(
                        _tensors, batch_first=True, padding_value=padding_value).transpose(-1, -2),
                )
            else:
                return_list.append(torch.stack(_tensors, dim=0))
        return tuple(return_list)

    def __len__(self):
        return len(self.example_list)


def evaluate(args, eval_dataset, model, tokenizer, global_step,
             is_saving_pred=False, file_prefix=""):
    logging.info("***** Running evaluation at {}*****".format(global_step))
    logging.info("  Num examples = %d", len(eval_dataset))
    logging.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = setup_eval_step(
        args, eval_dataset, collate_fn=eval_dataset.data_collate_fn,)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    _idx_ex = 0
    _eval_predict_data = []
    preds_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = [_t.to(args.device) for _t in batch]
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, token_type_ids=segment_ids,
                attention_mask=input_mask, labels=label_ids)
            tmp_eval_loss, logits = outputs[:2]
            # probs = torch.softmax(logits, dim=-1)  # nn.functional

        logits = logits.detach().cpu().numpy()
        # probs = probs.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy_np(logits, label_ids)
        preds = np.argmax(logits, axis=-1)  # bn
        preds_list.append(preds)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              # 'global_step': global_step,
              # 'loss': tr_loss / (global_step if global_step > 0 else 1)
              }

    qid_list = eval_dataset.get_all_qid()

    if is_saving_pred:
        outputs = np.concatenate(preds_list)
        assert len(qid_list) == outputs.size
        with open(os.path.join(args.output_dir, file_prefix + "saved_predictions.csv"),
                  "w", encoding="utf-8") as wfp:
            for _pred, _qid in zip(outputs, qid_list):
                wfp.write("{},{}{}".format(
                    _qid, chr(int(_pred) + ord("A")),
                    os.linesep
                ))
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
        args, train_dataset, collate_fn=train_dataset.data_collate_fn)
    model, optimizer, scheduler = setup_opt(args, model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    global_step = 0
    best_accu = 0.
    ma_dict = MovingAverageDict()
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _idx_epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration-{}({})".format(_idx_epoch, args.gradient_accumulation_steps),
                              disable=args.local_rank not in [-1, 0])
        step_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(
                input_ids=input_ids, token_type_ids=segment_ids,
                attention_mask=input_mask, labels=label_ids
            )
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

                if eval_dataset is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    cur_accu = evaluate(args, eval_dataset, model, tokenizer, global_step=global_step)
                    if cur_accu > best_accu:
                        best_accu = cur_accu
                        save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
        # evaluation each epoch or last epoch
        if (_idx_epoch == int(args.num_train_epochs) - 1) or (eval_dataset is not None and args.eval_steps <= 0):
            cur_accu = evaluate(args, eval_dataset, model, tokenizer, global_step=global_step)
            if cur_accu > best_accu:
                best_accu = cur_accu
                save_model_with_default_name(args.output_dir, model, tokenizer, args_to_save=args)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    with open(os.path.join(args.output_dir, "best_eval_results.txt"), "w") as fp:
        fp.write("{}{}".format(best_accu, os.linesep))


def main():
    parser = argparse.ArgumentParser()
    # data related
    parser.add_argument("--dataset", default="cqa", type=str, help="[cqa|race]")
    parser.add_argument("--model_class", default="roberta", type=str, help="[roberta|bert]")
    parser.add_argument("--data_dir", required=True, type=str, help="")
    parser.add_argument("--data_split", default="rand", type=str, help="The input data dir.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")

    define_hparams_training(parser)
    args = parser.parse_args()

    setup_prerequisite(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.model_class == "roberta":
        config_class = RobertaConfig
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMultipleChoice
    elif args.model_class == "bert":
        config_class = BertConfig
        tokenizer_class = BertTokenizer
        model_class = BertForMultipleChoice
    else:
        raise KeyError(args.model_class)

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    train_dataset = CommonsenseqaDataset("train", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)
    dev_dataset = CommonsenseqaDataset("dev", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)
    test_dataset = CommonsenseqaDataset("test", args.data_dir, tokenizer, args.do_lower_case, args.max_seq_length)

    if args.do_train:
        train(args, train_dataset, model, tokenizer, dev_dataset)

    if args.do_train and args.do_eval:  # load the best model
        model = model_class.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

    if args.do_eval:
        dev_accu = evaluate(args, dev_dataset, model, tokenizer, is_saving_pred=True, global_step=None, file_prefix="dev_")
        test_accu = evaluate(args, test_dataset, model, tokenizer, is_saving_pred=True, global_step=None, file_prefix="aux_")
        with open(os.path.join(args.output_dir, "predicted_results.txt"), "w") as fp:
            fp.write("{},{}{}".format(dev_accu, test_accu, os.linesep))


if __name__ == '__main__':
    main()