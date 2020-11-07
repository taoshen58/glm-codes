import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate, _use_shared_memory

import numpy as np
import sys
import os
from os.path import join
from tqdm import tqdm, trange
import json
import math
import random
import networkx as nx
import collections
import logging
import regex as re
import time
import copy
from configs import USER_HOME
from data_proc.concept_extractor import ConceptExtractor, NUM_CONCEPT_NET_REL, lemma_dict_path, \
    load_conceptnet_en_map_dump, conceptnet_dump_iter, conceptnet_en_path, REDUNDANT_RELATIONS, \
    load_list_from_file, conceptnet_rel2idx_out_path, triplet2sent_path, UNDIRECTED_RELATIONS
from utils.common import file_exists, dir_exists, load_json, save_json, parse_span_str, get_data_path_list, \
    get_statistics_for_num_list, load_pickle, save_pickle, save_jsonl_with_offset, load_jsonl_with_offset, \
    save_pkll_with_offset, load_pkll_with_offset
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists
from fuzzywuzzy import fuzz
from data_proc.index_cn import load_sent_index_offset, load_sent_index, load_sent_from_shard, get_data_type, \
    load_neighbor_cididxs, load_neighbor_cididxs_offsets, load_conceptnet, load_conceptnet_graph, \
    load_stop_ctkidx_list

MAX_NODE_CTX_LEN = 18


class DatasetLine(Dataset):
    SENT_CORPUS = ["gen", "omcs", "arc"]
    PARA_COPUS = ["wikipedia"]
    DATA_TYPE_LIST = SENT_CORPUS + PARA_COPUS

    def _negative_sampling_adv(self, src_ctkidx, tgt_ctkidxs=None):
        tgt_ctkidxs = tgt_ctkidxs or set()
        tgt_ctkidx_set = set(copy.copy(tgt_ctkidxs))
        if src_ctkidx in tgt_ctkidx_set:
            tgt_ctkidx_set.remove(src_ctkidx)
        tgt_ctkidx_list = list(tgt_ctkidx_set)

        assert self.cididx2ctkidx is not None  # and ...
        src_ctk = self.ctk_list[src_ctkidx]

        sampled_flag = False
        sampled_ctkidx = None
        sample_list, weight_list = None, None
        while not sampled_flag:
            _prob = random.random()
            if _prob < 0.1 or len(tgt_ctkidx_list) == 0:  # random sample
                sampled_ctkidx = self._negative_sampling(src_ctkidx)
                sampled_flag = True
            elif _prob < 0.2:
                if len(tgt_ctkidx_list) > 0:
                    sampled_ctkidx = random.choice(tgt_ctkidx_list)
                    sampled_flag = True
            else:
                if sample_list is None and weight_list is None:
                    # 1-hop and n-hop relations
                    hop1_candidates, hopn_candidates = [], []
                    for _head2tails, _tail2heads in [
                        (self.subj2objs, self.obj2subjs), (self.obj2subjs, self.subj2objs)]:
                        _rel2ctkidxs = _head2tails[src_ctkidx]
                        for _rel, _ctkidxs in _rel2ctkidxs.items():
                            _ctkidx_set = set(_ctkidxs)
                            # 1-hop
                            _ad_ctkidx_set = set(_ctkidx_set) & tgt_ctkidx_set
                            for _ad_ctkidx in _ad_ctkidx_set:
                                for _sb_ctkidx in _tail2heads[_ad_ctkidx][_rel]:
                                    hop1_candidates.append(_sb_ctkidx)
                            # multi-hop relations
                            _nb_ctkidx_set = tgt_ctkidx_set - set(_ctkidx_set)
                            for _nb_ctkidx in _nb_ctkidx_set:
                                for _sm_ctkidx in _tail2heads[_nb_ctkidx][_rel]:
                                    hopn_candidates.append(_sm_ctkidx)
                    candidates_list = [hop1_candidates, hopn_candidates]
                    sample_list, weight_list = [], []
                    for _candidates in candidates_list:
                        _lc_sample_list, _lc_weight_list = [], []
                        _cand2freq = collections.Counter(_candidates)
                        if src_ctkidx in _cand2freq:
                            _cand2freq.pop(src_ctkidx)
                        for _cand, _freq in _cand2freq.items():
                            _lc_sample_list.append(_cand)
                            _lc_weight_list.append(_freq)
                        if len(_lc_sample_list) == 0:
                            continue
                        _total_wei = sum(_lc_weight_list) * float(len(candidates_list))
                        _lc_weight_list = [_e/_total_wei for _e in _lc_weight_list]
                        sample_list.extend(_lc_sample_list)
                        weight_list.extend(_lc_weight_list)
                if len(sample_list) > 0:
                    sampled_ctkidx = random.choices(sample_list, weight_list)[0]
                    sampled_flag = True

            # filtering: if similar to src_ctkidx, re-sampling
            if sampled_flag:
                if sampled_ctkidx == src_ctkidx:
                    sampled_flag = False
                sampled_ctk = self.ctk_list[sampled_ctkidx]
                if fuzz.ratio(sampled_ctk, src_ctk) > 60:
                    sampled_flag = False
        return sampled_ctkidx

    def _negative_sampling(self, src_ctkidx, method="simple"):
        assert self.cididx2ctkidx is not None  # and ...
        src_ctk = self.ctk_list[src_ctkidx]
        # simplest version
        assert method == "simple"
        sampled_flag = False
        sampled_ctkidx = None
        while not sampled_flag:
            prob = random.random()
            if prob < 0.9:
                prob /= 0.9
                if prob < 0.5:  # out property
                    _head2tail = self.subj2objs
                    _rel2head = self.rel2subjs
                else: # in property
                    _head2tail = self.obj2subjs
                    _rel2head = self.rel2objs

                _rels = set(_head2tail[src_ctkidx].keys())
                _rels = _rels - REDUNDANT_RELATIONS
                if "/r/RelatedTo" in _rels:
                    _rels.remove("/r/RelatedTo")
                if len(_rels) > 0:
                    _rmd_rel = random.choice(list(_rels))
                    if len(_rel2head[_rmd_rel]) > 0:
                        sampled_ctkidx = random.choice(_rel2head[_rmd_rel])
                        sampled_flag = True
            else:
                sampled_ctkidx = self.ctk2idx[random.choice(self.ctk_list)]
                sampled_flag = True

            # filtering: if similar to src_ctkidx, re-sampling
            if sampled_flag:
                if sampled_ctkidx == src_ctkidx:
                    sampled_flag = False
                sampled_ctk = self.ctk_list[sampled_ctkidx]
                if fuzz.ratio(sampled_ctk, src_ctk) > 60:
                    sampled_flag = False
        return sampled_ctkidx

    def _load_negative_sampling_graph(self):  # this is ctk level !!!
        # self.subj2objs, self.obj2subjs, self.rel2subjs, self.rel2objs = None, None, None, None
        print("loading negative sampling graph")
        self.subj2objs = [collections.defaultdict(list) for _ in range(len(self.ctk_list))]  # out
        self.obj2subjs = [collections.defaultdict(list) for _ in range(len(self.ctk_list))]  # in
        self.rel2subjs = collections.defaultdict(list)
        self.rel2objs = collections.defaultdict(list)
        for _row in tqdm(conceptnet_dump_iter(conceptnet_en_path)):
            _id1, _id2 = _row[2], _row[3]
            _cididx1, _cididx2 = self.cid2idx[_id1], self.cid2idx[_id2]
            _ctkidx1, _ctkidx2 = self.cididx2ctkidx[_cididx1], self.cididx2ctkidx[_cididx2]
            _rel = _row[1]
            self.subj2objs[_ctkidx1][_rel].append(_ctkidx2)
            self.obj2subjs[_ctkidx2][_rel].append(_ctkidx1)
            self.rel2subjs[_rel].append(_ctkidx1)
            self.rel2objs[_rel].append(_ctkidx2)
            if _rel in UNDIRECTED_RELATIONS:
                self.subj2objs[_ctkidx2][_rel].append(_ctkidx1)
                self.obj2subjs[_ctkidx1][_rel].append(_ctkidx2)
                self.rel2subjs[_rel].append(_ctkidx2)
                self.rel2objs[_rel].append(_ctkidx1)
        print("\tDone")

    @classmethod
    def build_training_and_dev_datasets(cls, *args, **kwargs):
        train_dataset = cls(*args, **kwargs)
        dev_dataset = copy.copy(train_dataset)

        example_list = train_dataset.example_list

        dev_proportion = kwargs.get("dev_proportion") or 10000
        num_examples = len(example_list)

        idxs_filename = train_dataset.dev_filename_template.format(
            ",".join(train_dataset.data_type_list),
            "sf{}_dp{}".format(train_dataset.sent_format, dev_proportion),
            "json")
        idx_filepath = join(train_dataset.cached_dir, idxs_filename)

        if file_exists(idx_filepath):
            logging.info("dev indices file exists, load and verify from {}".format(idx_filepath))
            loaded_num_examples, dev_idxs = load_json(idx_filepath)
            assert loaded_num_examples == num_examples
        else:
            logging.info("dev indices file doesn\'t exists, sample and save to {}".format(idx_filepath))
            # split
            if dev_proportion < 1:
                num_dev = math.floor(num_examples * dev_proportion)
            else:
                num_dev = int(dev_proportion)

            ex_indices = list(range(num_examples))
            random.shuffle(ex_indices)
            dev_idxs = ex_indices[:num_dev]
            # save devidxs
            save_json([num_examples, dev_idxs], idx_filepath)

        dev_idxs_set = set(dev_idxs)
        train_example_list = []
        dev_example_list = []
        for _idx_s in range(num_examples):
            if _idx_s in dev_idxs_set:
                dev_example_list.append(example_list[_idx_s])
            else:
                train_example_list.append(example_list[_idx_s])

        train_dataset.example_list = train_example_list
        dev_dataset.example_list = dev_example_list
        logging.info("num of training examples is {} and number of dev examples is {}".format(
            len(train_example_list), len(dev_example_list)
        ))

        # future: uncomment
        # dev_dataset.is_train = False

        # cache for dev
        cache_filename = train_dataset.dev_filename_template.format(
            ",".join(train_dataset.data_type_list),
            "sf{}_df{}_tk{}_dp{}_devcache".format(
                train_dataset.sent_format, train_dataset.data_format,
                train_dataset.tokenizer_type, dev_proportion),
            "pt")
        cache_filepath = join(train_dataset.cached_dir, cache_filename)
        if file_exists(cache_filepath):
            logging.info("dev cache file exists, load and verify from {}".format(cache_filepath))
            dev_cache = torch.load(cache_filepath)
        else:
            logging.info("dev indices file doesn\'t exists, sample and save to {}".format(cache_filepath))
            dev_cache = []
            for _i in tqdm(range(len(dev_dataset))):
                features = dev_dataset[_i]
                dev_cache.append(features)
            torch.save(dev_cache, cache_filepath)
        dev_dataset.dev_cache = dev_cache

        return train_dataset, dev_dataset

    def __init__(
            self, data_type_list, do_lower_case, tokenizer, max_seq_len, cached_dir,
            threshold_stop_ctk=0, num_parallels=35, mask_proportion=0.15,
            use_simple_neg=False, use_invalid_omcs=False, use_nongraph=False,
            tokenizer_type="roberta", sent_format="cat", data_format="cn",
            is_train=True,
            load_reindex=True,
            *args, **kwargs,
    ):
        self.data_type_list = [_e for _e in self.DATA_TYPE_LIST if _e in data_type_list]  # re-order
        self.do_lower_case = do_lower_case
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cached_dir = cached_dir

        self.threshold_stop_ctk = threshold_stop_ctk
        self.num_parallels = num_parallels
        self.mask_proportion = mask_proportion

        # effect in __getitem__
        # self.disable_gen = disable_gen
        self.use_simple_neg = use_simple_neg
        self.use_invalid_omcs = use_invalid_omcs
        self.use_nongraph = use_nongraph
        self.tokenizer_type = tokenizer_type
        self.sent_format = sent_format
        self.data_format = data_format

        # if self.tokenizer_type == "bert": # use uncased nodel
        #     assert self.do_lower_case
        assert self.threshold_stop_ctk == 0
        if self.data_format == "cn":
            assert not self.use_simple_neg
        assert not self.use_invalid_omcs
        assert self.sent_format in ["cat", "single"]
        assert self.data_format in ["cn", "wwm", "span", "cn_rdm"]

        self.is_train = is_train
        self.load_reindex = load_reindex

        self.max_context_len = 240

        # conceptnet basics related member variables
        self.ctk_list, self.cid_list, self.ctk2idx, self.cid2idx, self.cididx2ctkidx, self.ctkidx2cididxs = \
            load_conceptnet()

        # ids for special tokens
        self._sep_id, self._mask_id, self._bos_id, self._eos_id, self._cls_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.sep_token, self.tokenizer.mask_token,
             self.tokenizer.bos_token, self.tokenizer.eos_token,
             self.tokenizer.cls_token]
        )
        self._pad_id, = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token, ])

        assert len(self.data_type_list) > 0

        # sent data
        self.sent_part_idxs = [0, ]
        self.sent_index_offset_list = []
        for _data_type in data_type_list:
            _offset_list = load_sent_index_offset(_data_type, self.cached_dir)
            self.sent_index_offset_list.extend(_offset_list)
            self.sent_part_idxs.append(len(self.sent_index_offset_list))

        # neignbor data
        self.cididx2nboffset = load_neighbor_cididxs_offsets(self.cached_dir)

        self.stop_ctkidxs = {}
        self.stop_cididxs = {}
        if self.threshold_stop_ctk > 0:
            for _dt in self.data_type_list:
                self.stop_ctkidxs[_dt] = set(
                    load_stop_ctkidx_list(self.cached_dir, stop_prop=self.threshold_stop_ctk))
        elif self.threshold_stop_ctk == 0:
            for _dt in self.data_type_list:
                if _dt in ["gen", "omcs"]:
                    thresh_stop = 2
                elif _dt == "arc":
                    thresh_stop = 150
                elif _dt == "wikipedia":
                    thresh_stop = 250
                else:
                    raise AttributeError(_dt)
                self.stop_ctkidxs[_dt] = set(load_stop_ctkidx_list(self.cached_dir, stop_prop=thresh_stop))
        else:
            raise AttributeError(self.threshold_stop_ctk)
        for _key in self.stop_ctkidxs:
            self.stop_cididxs[_key] = set(
                _cididx for _ctkidx in self.stop_ctkidxs[_key] for _cididx in self.ctkidx2cididxs[_ctkidx])

        # meta data wrt data type, 1) graph data 2) sent len
        prefix = "DS-ts{}-".format(self.threshold_stop_ctk)
        self.meta_filename_template = prefix+ "meta-{}-{}.{}"  # _data_type, "data|offset", "suffix"
        self.example_filename_template = prefix + "ex-{}-{}.{}"  # data_types, "data", "pkl"
        self.dev_filename_template = prefix + "dev-{}-{}.{}"  # data_types, "data", "pkl"

        self.sentidx2metaoffset = self.build_meta_data()

        if self.sent_format == "cat":
            self.example_list = self.generate_long_contexts()
        elif self.sent_format == "single":
            single_example_path = join(self.cached_dir, self.example_filename_template.format(
                ",".join(self.data_type_list), "single_sent", "pkl"))
            if file_exists(single_example_path):
                self.example_list = load_pickle(single_example_path)
            else:
                print("did not find single exmaples, building...")
                self.example_list = []
                for _sentidx, _sentoffset in enumerate(tqdm(self.sent_index_offset_list)):
                    _data_type = get_data_type(_sentidx, self.sent_part_idxs, self.data_type_list)
                    _meta_data = self._load_meta(_sentidx, _data_type)
                    if self.is_valid_sent(_meta_data, _data_type):
                        self.example_list.append([_sentidx])
                save_pickle(self.example_list, single_example_path)
        else:
            raise KeyError(self.sent_format)

        # filtering
        # print("Filtering too short examples")
        # new_example_list = []
        # for example in tqdm(self.example_list):
        #     ex_len = 0
        #     must_keep = False
        #     for _sentidx in example:
        #         _data_type = get_data_type(_sentidx, self.sent_part_idxs, self.data_type_list)
        #         if _data_type == "omcs":
        #             must_keep = True
        #             break
        #         _meta = self._load_meta(_sentidx, _data_type)
        #         ex_len += _meta[1]
        #     if must_keep or ex_len > self.max_context_len // 2:
        #         new_example_list.append(example)
        # print("\tfiltering completed: # before {}, after {}".format(len(self.example_list), len(new_example_list)))
        # self.example_list = new_example_list

        self.subj2objs, self.obj2subjs, self.rel2subjs, self.rel2objs = None, None, None, None
        if self.data_format in ["cn", "cn_rdm"]:
            self._load_negative_sampling_graph()

        # cache for consistent evaluation
        self.dev_cache = None

    def _load_meta(self, sentidx, data_type):
        if data_type is None:
            data_type = get_data_type(sentidx, self.sent_part_idxs, self.data_type_list)

        return load_pkll_with_offset(
            self.sentidx2metaoffset[sentidx],
            join(self.cached_dir, self.meta_filename_template.format(data_type, "data", "pkll"))
        )

    def split_to_token(self, cat_senttext):
        tgt_special_tokens = self.tokenizer.all_special_tokens

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                sub_text = sub_text.strip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text, tokenize_fn):
            if not text:
                return []
            if not tok_list:
                return re.findall(self.tokenizer.pat, cat_senttext)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in tgt_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return sum((tokenize_fn(token) if token not in tgt_special_tokens \
                            else [token] for token in tokenized_text), [])

        if self.tokenizer_type == "bert":
            bert_tokenzie_fn = self.tokenizer.basic_tokenizer.tokenize
            sent_token_list = split_on_tokens(tgt_special_tokens, cat_senttext, bert_tokenzie_fn)
        elif self.tokenizer_type == "roberta":
            roberta_tokenize_fn = lambda _arg: re.findall(self.tokenizer.pat, _arg)
            sent_token_list = split_on_tokens(tgt_special_tokens, cat_senttext, roberta_tokenize_fn)
        else:
            raise KeyError(self.tokenizer_type)
        return sent_token_list

    def __getitem__(self, item):
        if self.dev_cache is not None and isinstance(self.dev_cache, list):
            if self.data_format in ["span", "wwm"]:
                return self.dev_cache[item][0], self.dev_cache[item][1], self.dev_cache[item][-1]
            else:
                return tuple(self.dev_cache[item])

        sentidx_list = self.example_list[item]  # example
        sentdp_list = [ get_data_type(_sentidx, self.sent_part_idxs, self.data_type_list)
                        for _sentidx in sentidx_list]
        meta_list = [
            self._load_meta(_sentidx, _sentdp)
            for _sentidx, _sentdp in zip(sentidx_list, sentdp_list)
        ]
        ins_ctkidxs_list = [set(_meta[0]) for _meta in meta_list]
        sentdata_list = [load_sent_from_shard(self.sent_index_offset_list[_sentidx], self.cached_dir, _sentdp)
                         for _sentidx, _sentdp in zip(sentidx_list, sentdp_list)]

        # concatenate all sentence [HOW to identify difference key]
        cat_ctk2spans = dict()
        cat_senttext = self.tokenizer.cls_token + " "
        for _idx_s, _sentdata in enumerate(sentdata_list):
            _sentid = _sentdata[0]
            _senttext = _sentdata[1]
            _ctk2spans = _sentdata[2]

            if self.tokenizer_type == "bert" and self.do_lower_case:
                _senttext = _senttext.lower()

            # upadte _ctk2spans
            for _key, _spans in _ctk2spans.items():
                if _key.startswith("/c/"):  # should be /c/en/
                    _ctk_pre = _key.split("/")[3]
                    _ctk = " ".join(_ctk_pre.split("_"))
                else:
                    _ctk = _key
                if _ctk not in cat_ctk2spans:
                    cat_ctk2spans[_ctk] = list()
                for _span in _spans:
                    cat_ctk2spans[_ctk].append([_span[0] + len(cat_senttext), _span[1] + len(cat_senttext)])
            if sentdp_list[_idx_s] in self.SENT_CORPUS and _idx_s < len(sentdata_list)-1:
                if self.tokenizer_type == "roberta":
                    sep_str = " {} {} ".format(self.tokenizer.cls_token, self.tokenizer.sep_token)
                elif self.tokenizer_type == "bert":
                    sep_str = " {} ".format(self.tokenizer.sep_token)
                else:
                    raise KeyError(self.tokenizer_type)
                cat_senttext += (_senttext + sep_str)
            else:
                cat_senttext += (_senttext + " ")

        if self.data_format == "wwm":
            sent_token_list = self.split_to_token(cat_senttext)
            mask_dict = {}
            for _idx_tk, _tk in enumerate(sent_token_list):
                if _tk in self.tokenizer.all_special_tokens:
                    continue
                prob = random.random()
                if prob < self.mask_proportion:
                    prob /= self.mask_proportion
                    if prob < 0.8:  # 80% randomly change token to mask token
                        mask_dict[_idx_tk] = 1
                    elif prob < 0.9:  # 10% randomly change token to random token
                        mask_dict[_idx_tk] = 2
                    else:  # -> rest 10% randomly keep current token
                        mask_dict[_idx_tk] = 3
            wp_list, id_list, pos_list = self._continue_tokenize_for_wordpiece(sent_token_list)
            input_ids, label_ids = [], []
            for _wp, _id, _pos in zip(wp_list, id_list, pos_list):
                if _pos in mask_dict:
                    label_ids.append(_id)
                    if mask_dict[_pos] == 1:
                        input_ids.append(self._mask_id)
                    elif mask_dict[_pos] == 2:
                        _rdm_tk = self._random_choice_token_from_vocab()
                        _rdm_id = self.tokenizer.convert_tokens_to_ids([_rdm_tk, ])[0]
                        input_ids.append(_rdm_id)
                    else:  # == 3
                        input_ids.append(_id)
                else:
                    input_ids.append(_id)
                    label_ids.append(-1)
            if len(input_ids) > self.max_seq_len - 1:
                input_ids = input_ids[:(self.max_seq_len - 1)]
                label_ids = label_ids[:(self.max_seq_len - 1)]
            input_ids.append(self._sep_id)
            label_ids.append(-1)
            mask_ids = [1] * len(label_ids)
            return torch.tensor(input_ids), torch.tensor(mask_ids), torch.tensor(label_ids),

        elif self.data_format == "span":
            # this is similar to xxx
            sent_token_list = self.split_to_token(cat_senttext)
            budget_len = math.ceil(len(sent_token_list) * self.mask_proportion)
            aux_token_idxs_list = [[_idx_tk] for _idx_tk, _tk in enumerate(sent_token_list)
                                   if _tk in self.tokenizer.all_special_tokens]
            token_idxs_list = []
            while budget_len > 0:
                _unused_tk_idx2len = self._get_unused_idxs_and_lens(len(sent_token_list),
                                                                    token_idxs_list + aux_token_idxs_list)
                if len(_unused_tk_idx2len) == 0:
                    break
                _rdm_tk_idx = random.choice(list(_unused_tk_idx2len.keys()))
                _rdm_max_len = _unused_tk_idx2len[_rdm_tk_idx]
                _rdm_len = self._geometric_sampling(0.2, _rdm_max_len)
                _rdm_len = min(_rdm_len, budget_len)
                _rdm_tk_idxs = list(range(_rdm_tk_idx, _rdm_tk_idx + _rdm_len))
                if budget_len - len(_rdm_tk_idxs) >= 0:
                    budget_len -= len(_rdm_tk_idxs)
                    token_idxs_list.append(_rdm_tk_idxs)
            input_ids, label_ids, org_ids, pos_list, wp_list \
                = self._masking_wrt_idxs_list(sent_token_list, token_idxs_list, self.max_seq_len, with_bos=True)
            mask_ids = [1] * len(label_ids)
            return torch.tensor(input_ids), torch.tensor(mask_ids), torch.tensor(label_ids),

        ctks_graph = set()
        for _ins_ctkidxs in ins_ctkidxs_list:
            for _ctkidx in _ins_ctkidxs:
                _ctk = self.ctk_list[_ctkidx]
                assert _ctk in cat_ctk2spans
                ctks_graph.add(_ctk)
        ctks_nongraph = set()
        for _ctk in cat_ctk2spans.keys():
            if _ctk not in ctks_graph:
                ctks_nongraph.add(_ctk)

        if self.data_format == "cn_rdm":
            # modify `ins_ctkidxs_list` to include all node (include stop one)
            # ins_ctkidxs_list = []
            # for _idx_s, _sentdata in enumerate(sentdata_list):
            #     _ctk2spans = _sentdata[2]
            #     _ins_ctkidxs = set()
            #     for _ctk in _ctk2spans:
            #         assert not _ctk.startswith("/c/")
            #         _ins_ctkidxs.add(self.ctk2idx[_ctk])
            #     ins_ctkidxs_list.append(_ins_ctkidxs)
            ctks_graph = set(_ctk for _ctk in cat_ctk2spans)
            ctks_nongraph = set()

        # ++++ tokenizer ++++

        sent_token_list = self.split_to_token(cat_senttext)
        charidx2tokenidx = self._parse_tk_idx_list_wrt_char(cat_senttext, sent_token_list)

        # mask sampling
        prob_thresh = self.mask_proportion
        budget_len = math.ceil(len(sent_token_list) * prob_thresh)
        budget_neg_len = - math.floor(len(sent_token_list) * (0.3 - prob_thresh))

        gctk_list = list(ctks_graph)
        octk_list = list(ctks_nongraph)

        aux_token_idxs_list = [[_idx_tk] for _idx_tk, _tk in enumerate(sent_token_list)
                               if _tk in self.tokenizer.all_special_tokens]
        token_idxs_list = []
        masked_gctks = set()
        masked_ngctks = set()
        while budget_len > 0:
            prob = random.random()
            if self.data_format == "cn":
                thresh_for_gctk = 0.5
            elif self.data_format == "cn_rdm":
                thresh_for_gctk = 0.8
                assert len(octk_list) == 0
            else:
                raise KeyError(self.data_format)

            if prob < thresh_for_gctk:  # sample from graph
                while len(gctk_list) > 1:
                    _rdm_idx = random.choice(range(len(gctk_list)))
                    _rdm_ctkidx = gctk_list.pop(_rdm_idx)
                    _cspans = cat_ctk2spans[_rdm_ctkidx]
                    tkidxs_list = []
                    for _cspan in _cspans:
                        _tkidxs = list(range(charidx2tokenidx[_cspan[0]], charidx2tokenidx[_cspan[1] - 1] + 1))
                        tkidxs_list.append(_tkidxs)
                    _new_idx_set = self._new_idxs_for_idxs_list(token_idxs_list, tkidxs_list)
                    if budget_len - len(_new_idx_set) >= budget_neg_len:
                        budget_len -= len(_new_idx_set)
                        token_idxs_list.extend(tkidxs_list)
                        masked_gctks.add(_rdm_ctkidx)
                        break
            elif prob < 0.8:  # sample from non-graph
                while len(octk_list) > 0:
                    _rdm_idx = random.choice(range(len(octk_list)))
                    _rdm_ctkidx = octk_list.pop(_rdm_idx)
                    _cspan = random.choice(cat_ctk2spans[_rdm_ctkidx])
                    tkidxs_list = [list(range(charidx2tokenidx[_cspan[0]], charidx2tokenidx[_cspan[1] - 1] + 1))]
                    _new_idx_set = self._new_idxs_for_idxs_list(token_idxs_list, tkidxs_list)
                    if len(_new_idx_set) == 0:
                        continue
                    if budget_len - len(_new_idx_set) >= budget_neg_len:
                        budget_len -= len(_new_idx_set)
                        token_idxs_list.extend(tkidxs_list)
                        masked_ngctks.add(_rdm_ctkidx)
                        break
            else:  # random sampling
                _unused_tk_idx2len = self._get_unused_idxs_and_lens(len(sent_token_list),
                                                                    token_idxs_list+aux_token_idxs_list)
                if len(_unused_tk_idx2len) == 0:
                    break
                _rdm_tk_idx = random.choice(list(_unused_tk_idx2len.keys()))
                _rdm_max_len = _unused_tk_idx2len[_rdm_tk_idx]
                _rdm_len = self._geometric_sampling(0.2, _rdm_max_len)
                _rdm_len = min(_rdm_len, budget_len)
                _rdm_tk_idxs = list(range(_rdm_tk_idx, _rdm_tk_idx + _rdm_len))
                if budget_len - len(_rdm_tk_idxs) >= budget_neg_len:
                    budget_len -= len(_rdm_tk_idxs)
                    token_idxs_list.append(_rdm_tk_idxs)
        # token_idxs_list.pop(0)

        # all these outputs are truncated
        input_ids, label_ids, org_ids, pos_list, wp_list \
            = self._masking_wrt_idxs_list(sent_token_list, token_idxs_list, self.max_seq_len, with_bos=True)

        # return the graph's nodes
        tkidx2wpidxs = collections.defaultdict(list)
        for _wp_idx, _tk_idx in enumerate(pos_list):
            tkidx2wpidxs[_tk_idx].append(_wp_idx)
        max_pos = max(pos_list)

        node_se_list = []  # n, 2
        node_ctk_list = []
        for _ctk in cat_ctk2spans:
            _spans = cat_ctk2spans[_ctk]
            for _span in _spans:
                _start_tkidx = charidx2tokenidx[_span[0]]
                _end_tkidx = charidx2tokenidx[_span[1] - 1]
                if _start_tkidx > max_pos:  # constrain by max len
                    continue
                _end_tkidx = min(_end_tkidx, max_pos)
                # from tkidx2wpidx
                _start_wpidx = min(tkidx2wpidxs[_start_tkidx])
                _end_wpidx = max(tkidx2wpidxs[_end_tkidx])
                node_se_list.append([_start_wpidx, _end_wpidx])
                node_ctk_list.append(_ctk)

        # node2ses
        max_node_num = self.max_seq_len // 10
        max_num_ses_per_node = 4
        node_ctk2se_list = collections.OrderedDict()
        node_ctx_pos_list, node_ctx_neg_list = [], []

        # # re-rankraise NotImplementedError(args.model_class)raise NotImplementedError(args.model_class)
        tmp_node_idxs_list = [[], [], []]
        for _idx_node, _node_ctk in enumerate(node_ctk_list):
            if _node_ctk in masked_gctks:
                tmp_node_idxs_list[0].append(_idx_node)
            elif _node_ctk in masked_ngctks:
                tmp_node_idxs_list[1].append(_idx_node)
            else:
                tmp_node_idxs_list[2].append(_idx_node)
        reranked_node_idxs = tmp_node_idxs_list[0] + tmp_node_idxs_list[1] + tmp_node_idxs_list[2]
        # # combine spans
        for _idx_node in reranked_node_idxs:
            _node_se = node_se_list[_idx_node]
            _node_ctk = node_ctk_list[_idx_node]
            _masked = int((_node_ctk in masked_gctks) or (self.use_nongraph and _node_ctk in masked_ngctks))
            if not _masked:  # remove non-masked node
                continue
            if _node_ctk not in node_ctk2se_list:
                if len(node_ctk2se_list) >= max_node_num:
                    continue
                node_ctk2se_list[_node_ctk] = [[], _masked]
                node_ctx_pos_list.append(_node_ctk)
            node_ctk2se_list[_node_ctk][0].append(_node_se)
        for _node in node_ctk2se_list:
            if len(node_ctk2se_list[_node][0]) > max_num_ses_per_node:
                random.shuffle(node_ctk2se_list[_node][0])
                node_ctk2se_list[_node][0] = node_ctk2se_list[_node][0][:max_num_ses_per_node]
        # neg sampling
        ctkidxs_graph = set([self.ctk2idx[_ctk] for _ctk in ctks_graph])
        for _node_ctk in node_ctx_pos_list:
            _ctkidx = self.ctk2idx[_node_ctk]
            if self.use_simple_neg:
                _sampled_ctk = self.ctk_list[self._negative_sampling(_ctkidx)]
            else:
                _sampled_ctk = self.ctk_list[self._negative_sampling_adv(_ctkidx, ctkidxs_graph)]
            node_ctx_neg_list.append(_sampled_ctk)

        # tokenizing and digitizing for pos and neg
        node_ctx_tokens_list = []
        for _node_ctk in node_ctx_pos_list + node_ctx_neg_list:
            _node_ctk = "Node: " + _node_ctk
            if self.tokenizer_type == "bert" and self.do_lower_case:
                _node_ctk = _node_ctk.lower()
            _ctk_ctx = self.tokenizer.cls_token + " " + _node_ctk
            _node_ctx_tokens = self.tokenizer.tokenize(_ctk_ctx)
            if len(_node_ctx_tokens) > MAX_NODE_CTX_LEN-1:
                _node_ctx_tokens = _node_ctx_tokens[:(MAX_NODE_CTX_LEN-1)]
            _node_ctx_tokens.append(self.tokenizer.sep_token)
            node_ctx_tokens_list.append(_node_ctx_tokens)

        # max_len
        max_num_ses = 0
        max_len_ctx_tks = 0
        for _se_list, _masked in node_ctk2se_list.values():
            max_num_ses = max(max_num_ses, len(_se_list))
        for _node_ctx_tokens in node_ctx_tokens_list:
            max_len_ctx_tks = max(max_len_ctx_tks, len(_node_ctx_tokens))

        # convert to pytorch's input
        # # [n,max_num_ses,2], [n]
        node_ses = []
        node_masked_flag = []
        node_ctx_ids = []

        for _se_list, _masked in node_ctk2se_list.values():
            node_ses.append(_se_list + [[-1, -1] for _ in range(max_num_ses - len(_se_list))])
            node_masked_flag.append(_masked)
        for _node_ctx_tokens in node_ctx_tokens_list:
            node_ctx_ids.append(
                self.tokenizer.convert_tokens_to_ids(_node_ctx_tokens) +
                [self._pad_id] * (max_len_ctx_tks - len(_node_ctx_tokens))
            )
        # # handle emptyd data
        if len(node_ses) == 0 and len(node_masked_flag) == 0:
            node_ses = [[[-1, -1]]]  # 1,1,2
            node_masked_flag = [-1]
        if len(node_ctx_ids) == 0:
            node_ctx_ids = [[self._pad_id], [self._pad_id]]  # 2,1

        node_ctx_pos_ids, node_ctx_neg_ids = node_ctx_ids[:len(node_ctx_ids) // 2], node_ctx_ids[
                                                                                    len(node_ctx_ids) // 2:]

        # return example
        mask_ids = [1] * len(input_ids)

        return_list = [
            torch.tensor(input_ids),
            torch.tensor(mask_ids),

            torch.tensor(node_ctx_pos_ids),
            torch.tensor(node_ctx_neg_ids),
            torch.tensor(node_ses),
            torch.tensor(node_masked_flag),

            torch.tensor(label_ids),
        ]
        return tuple(return_list)

    def data_collate_fn(self, batch):
        tensors_list = list(zip(*batch))
        return_list = []

        for _idx_t, _tensors in enumerate(tensors_list):
            if _idx_t != (len(tensors_list)-1) and _idx_t in [0, 2, 3]:  # for input_ids
                padding_value = self._pad_id
            elif _idx_t == 1:  # for rep_mask
                padding_value = 0
            else:
                padding_value = -1
            if _idx_t != (len(tensors_list)-1) and _idx_t in [2, 3, 4]:  # graph node
                # 2D padding
                _max_len_last_dim = 0
                for _tensor in _tensors:
                    _max_len_last_dim = max(_max_len_last_dim, _tensor.size(1))
                # padding
                _new_tensors = []
                for _tensor in _tensors:
                    _pad_shape = list(_tensor.size())
                    _pad_shape[1] = _max_len_last_dim - _tensor.size(1)
                    _pad_tensor = torch.full(_pad_shape, padding_value, device=_tensor.device, dtype=_tensor.dtype)
                    _new_tensor = torch.cat([_tensor, _pad_tensor], dim=1)
                    _new_tensors.append(_new_tensor)
            else:
                _new_tensors = _tensors
            return_list.append(
                torch.nn.utils.rnn.pad_sequence(_new_tensors, batch_first=True, padding_value=padding_value),
            )
        return return_list

    @classmethod
    def batch2feed_dict(cls, batch, data_format):
        if data_format in ["cn", "cn_rdm"]:
            inputs = {'input_ids': batch[0],  # bs, sl
                      'attention_mask': batch[1],  #
                      'node_ctx_pos_ids': batch[2],
                      'node_ctx_neg_ids': batch[3],
                      'node_ses': batch[4],
                      'node_masked_flag': batch[5],
                      'masked_lm_labels': batch[-1]}
        elif data_format in ["wwm", "span"]:
            inputs = {'input_ids': batch[0],  # bs, sl
                      'attention_mask': batch[1],  #
                      'masked_lm_labels': batch[-1]}
        else:
            raise KeyError(data_format)
        return inputs

    def __len__(self):
        return len(self.example_list)

    def _continue_tokenize_for_wordpiece(self, token_list):
        wp_list = []
        pos_list = []
        if self.tokenizer_type == "bert":
            for _idx_tk, _tk in enumerate(token_list):
                _wps = self.tokenizer.tokenize(_tk)
                wp_list.extend(_wps)
                pos_list.extend([_idx_tk] * len(_wps))
        elif self.tokenizer_type == "roberta":
            for _idx_tk, _tk in enumerate(token_list):
                if _tk in self.tokenizer.all_special_tokens:
                    _wps = [_tk]
                else:
                    if sys.version_info[0] == 2:
                        _tk = ''.join(self.tokenizer.byte_encoder[ord(b)] for b in _tk)
                    else:
                        _tk = ''.join(self.tokenizer.byte_encoder[b] for b in _tk.encode('utf-8'))
                    _wps = [bpe_token for bpe_token in self.tokenizer.bpe(_tk).split(' ')]
                wp_list.extend(_wps)
                pos_list.extend([_idx_tk] * len(_wps))
        else:
            raise NotImplementedError(self.tokenizer_type)

        id_list = self.tokenizer.convert_tokens_to_ids(wp_list)
        assert len(wp_list) == len(id_list)
        return wp_list, id_list, pos_list

    def _random_choice_token_from_vocab(self):
        if self.tokenizer_type == "bert":
            return random.choice(list(self.tokenizer.vocab.items()))[0]
        elif self.tokenizer_type == "roberta":
            return random.choice(list(self.tokenizer.encoder.items()))[0]
        else:
            raise NotImplementedError(self.tokenizer_type)

    def _masking_wrt_idxs_list(self, sent_token_list, token_idxs_list, max_seq_len=None, with_bos=False):
        mask_dict = {}
        for _tk_idxs in token_idxs_list:
            prob = random.random()
            if prob < 0.8:  # 80% randomly change token to mask token
                _mask_type = 1
            elif prob < 0.9:  # 10% randomly change token to random token
                _mask_type = 1
            else:  # -> rest 10% randomly keep current token
                _mask_type = 1

            for _tk_idx in _tk_idxs:
                mask_dict[_tk_idx] = _mask_type

        wp_list, id_list, pos_list = self._continue_tokenize_for_wordpiece(sent_token_list)

        # prepare the data
        input_ids = []
        label_ids = []
        org_ids = []
        for _wp, _id, _pos in zip(wp_list, id_list, pos_list):
            org_ids.append(_id)
            if _pos in mask_dict:
                label_ids.append(_id)
                if mask_dict[_pos] == 1:
                    input_ids.append(self._mask_id)
                elif mask_dict[_pos] == 2:
                    _rdm_tk = self._random_choice_token_from_vocab()
                    _rdm_id = self.tokenizer.convert_tokens_to_ids([_rdm_tk, ])[0]
                    input_ids.append(_rdm_id)
                else:  # == 3
                    input_ids.append(_id)
            else:
                input_ids.append(_id)
                label_ids.append(-1)

        max_len = (max_seq_len - 1) if with_bos else (max_seq_len - 2)
        if max_seq_len is not None and len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            label_ids = label_ids[:max_len]
            org_ids = org_ids[:max_len]
            pos_list = pos_list[:max_len]

        input_ids = input_ids + [self._sep_id]
        label_ids = label_ids + [-1]
        org_ids = org_ids + [self._sep_id]
        if not with_bos:
            input_ids = [self._cls_id] + input_ids
            label_ids = [-1] + label_ids
            org_ids = [self._cls_id] + org_ids

        return input_ids, label_ids, org_ids, pos_list, wp_list

    def _parse_tk_idx_list_wrt_char(self, _text, _token_list):
        span_list = []
        if self.tokenizer_type == "bert":
            char_buffer, char_idxs = [], []
            _idx_tk = 0
            for _idx_c, _char in enumerate(_text):
                char_buffer.append(_char)
                char_idxs.append(_idx_c)
                _cur_str = "".join(char_buffer)
                _cur_list = self.split_to_token(_cur_str)
                try:
                    _cur_list.index(_token_list[_idx_tk])
                    span_list.append(char_idxs[0])
                    char_buffer, char_idxs = [], []
                    _idx_tk += 1
                except ValueError:
                    pass
                except IndexError:
                    break
            span_list.append(len(_text))
        elif self.tokenizer_type == "roberta":
            _tmp_find_idx = 0
            for _tk in _token_list:
                _found_idx = _text.find(_tk, _tmp_find_idx)
                _end_idx = _found_idx + len(_tk)
                assert _found_idx >= 0
                span_list.append(_found_idx)
                _tmp_find_idx = _end_idx
            span_list.append(len(_text))
        else:
            raise KeyError(self.tokenizer_type)
        if len(span_list) != len(_token_list) + 1:
            print(_text, _token_list)
            print(span_list)
        assert len(span_list) == len(_token_list) + 1
        # # build char2token
        tk_idx_list = []
        for _idx_tk, _char_start in enumerate(span_list[:-1]):
            _char_end = span_list[_idx_tk + 1]
            _token_act_len = _char_end - _char_start
            tk_idx_list.extend([_idx_tk] * _token_act_len)
        return tk_idx_list

    def build_meta_data(self):
        k_hop = 3

        def _processor(local_sent_offset_list, local_data_type):
            local_res_list = []
            for _sent_offset in tqdm(local_sent_offset_list, desc="in processor"):
                sent_data = load_sent_from_shard(_sent_offset, self.cached_dir, local_data_type)
                sent_id, sent_text, ctk2spans = sent_data[0], sent_data[1], sent_data[2]
                # 1) find all node -- graph_cididxs
                if local_data_type == "gen":
                    cid2spans = ctk2spans
                    graph_cididxs = set(self.cid2idx[_cid] for _cid in cid2spans)
                else:
                    ctkidxs = [self.ctk2idx[ctk] for ctk in ctk2spans]
                    ctkidxs = set(ctkidxs)  # - self.stop_ctkidxs[local_data_type]
                    cididxs = set(_cididx for _ctkidx in ctkidxs for _cididx in self.ctkidx2cididxs[_ctkidx])
                    graph_cididxs = set()
                    for _out_cididx in cididxs:
                        if _out_cididx in self.stop_cididxs[local_data_type]:
                            continue

                        _out_ctkidx = self.cididx2ctkidx[_out_cididx]
                        _tgt_cididxs = set()
                        for _ctkidx in ctkidxs:
                            if _ctkidx != _out_ctkidx:
                                _tgt_cididxs.update(self.ctkidx2cididxs[_ctkidx])

                        _nodes_list = load_neighbor_cididxs(_out_cididx, self.cididx2nboffset, self.cached_dir)[1:k_hop+1]
                        _node_set = set(_node for _nodes in _nodes_list for _node in _nodes)
                        _inter_nodes = _tgt_cididxs & _node_set
                        # _filtered_inter_nodes =
                        if len(_inter_nodes) > 0:
                            graph_cididxs.add(_out_cididx)
                            graph_cididxs.update(_inter_nodes - self.stop_cididxs[local_data_type])
                graph_ctkidxs = list(set([self.cididx2ctkidx[_cididx] for _cididx in graph_cididxs]))
                # 2. sent len
                sent_token_list = self.tokenizer.tokenize(sent_text)
                sent_len = len(sent_token_list) + 2
                local_res_list.append([graph_ctkidxs, sent_len])
            return local_res_list

        sentidx2metaoffset = []
        for _data_type in tqdm(self.data_type_list, desc="data_type"):
            _data_path = join(self.cached_dir, self.meta_filename_template.format(_data_type, "data", "pkll"))
            _offset_path = join(self.cached_dir, self.meta_filename_template.format(_data_type, "offset", "pkl"))
            if file_exists(_data_path) and file_exists(_offset_path):
                print("meta data for {} is found, loading".format(_data_type))
                _offsets = load_pickle(_offset_path)
            else:
                print("meta data for {} is not found, building".format(_data_type))
                _data_type_idx = self.data_type_list.index(_data_type)
                _sent_offset_list = self.sent_index_offset_list[
                                    self.sent_part_idxs[_data_type_idx]: self.sent_part_idxs[_data_type_idx+1]]
                _result_list = []
                _proc_buffer = []
                for _idx_so, _sent_offset in enumerate(tqdm(_sent_offset_list, desc="sents")):
                    _proc_buffer.append(_sent_offset)
                    if len(_proc_buffer) == 10000 * self.num_parallels or _idx_so == (len(_sent_offset_list) - 1):
                        if self.num_parallels == 1:
                            _res = _processor(_proc_buffer, _data_type)
                        else:
                            _res = combine_from_lists(
                                multiprocessing_map(
                                    _processor, dict_args_list=[
                                        {"local_sent_offset_list": _d, "local_data_type": _data_type}
                                        for _d in split_to_lists(_proc_buffer, self.num_parallels)
                                    ], num_parallels=self.num_parallels
                                ), ordered=True)
                        _result_list.extend(_res)
                        _proc_buffer = []
                assert len(_result_list) == len(_sent_offset_list)
                _offsets = save_pkll_with_offset(_result_list, _data_path)
                save_pickle(_offsets, _offset_path)
            sentidx2metaoffset.extend(_offsets)

        return sentidx2metaoffset

    def is_valid_sent(self, meta, data_type=None):
        _sent_ctkidxs, _sent_len = meta[0], meta[1]
        if len(_sent_ctkidxs) > 0 and _sent_len <= self.max_context_len - 2:
            if data_type is not None and data_type not in ["gen", "omcs"] and len(_sent_ctkidxs) < 2:
                return False
            else:
                return True
        else:
            return False

    def _enrich_sent_wrt_limit(
            self, ct_sentidx, max_context_len,
            valid_ctkidx2sentidxs, valid_sentidx2ctkidxs, valid_sentidx2len,
            banned_sentidx_set=None, ctkidxs_in_sent=None, use_acc=False
    ):
        if ctkidxs_in_sent is None:
            try:
                ctkidxs_in_sent = set(valid_sentidx2ctkidxs[ct_sentidx])
            except KeyError:
                ctkidxs_in_sent = set()
        ctkidxs_in_sent = list(_e for _e in ctkidxs_in_sent)
        random.shuffle(ctkidxs_in_sent)
        cand_sents = []
        for _ctkidx in ctkidxs_in_sent:
            try:
                _sentidxs = valid_ctkidx2sentidxs[_ctkidx]

            except KeyError:
                continue
            if _sentidxs is None:
                continue

            for _sentidx in _sentidxs:
                if _sentidx == ct_sentidx:
                    continue
                if banned_sentidx_set is not None:
                    if isinstance(banned_sentidx_set, set):
                        if _sentidx in banned_sentidx_set:
                            continue
                    elif isinstance(banned_sentidx_set, list):
                        if banned_sentidx_set[_sentidx]:
                            continue
                    else:
                        if any(_sentidx in _e for _e in banned_sentidx_set):
                            continue
                cand_sents.append(_sentidx)
                if use_acc and len(cand_sents) == 1000:
                    break

        # sampling
        sentidx2freq = collections.Counter(cand_sents)
        if ct_sentidx in sentidx2freq:
            sentidx2freq.pop(ct_sentidx)
        if banned_sentidx_set is not None:  # remove
            for _sentidx in list(sentidx2freq.keys()):
                if isinstance(banned_sentidx_set, set):
                    if _sentidx in banned_sentidx_set:
                        sentidx2freq.pop(_sentidx)
                elif isinstance(banned_sentidx_set, list):
                    if banned_sentidx_set[_sentidx]:
                        sentidx2freq.pop(_sentidx)
                else:
                    if any(_sentidx in _e for _e in banned_sentidx_set):
                        sentidx2freq.pop(_sentidx)

        sampled_sentidxs = []
        reamining_seq_len = max_context_len - valid_sentidx2len[ct_sentidx]
        while reamining_seq_len > 0 and len(sentidx2freq) > 0:
            _sample_list = []
            _weight_list = []
            for _sentidx, _freq in sentidx2freq.items():
                _sample_list.append(_sentidx)
                _weight_list.append(_freq)
            _rmd_sentdix = random.choices(_sample_list, _weight_list, k=1)[0]
            _new_sent_len = valid_sentidx2len[_rmd_sentdix]
            if reamining_seq_len - _new_sent_len >= 0:
                sampled_sentidxs.append(_rmd_sentdix)
                sentidx2freq.pop(_rmd_sentdix)
                reamining_seq_len -= _new_sent_len
            else:
                break
        return sampled_sentidxs

    def generate_long_contexts(self):
        # part 1: for ["gen", "omcs", "arc"]
        sent_data_type_list = [_data_type for _data_type in self.data_type_list if _data_type in self.SENT_CORPUS]
        sent_example_path = join(
            self.cached_dir, self.example_filename_template.format(
                ",".join(sent_data_type_list), "sent" if not self.use_invalid_omcs else "sent-invalid", "pkl"))

        if file_exists(sent_example_path):
            print("\texample pickle for {} is found, loading.".format(",".join(sent_data_type_list)))
            sent_example_list = load_pickle(sent_example_path)
        else:
            print("\texample pickle for {} is not found, building.".format(",".join(sent_data_type_list)))
            num_sents = self.sent_part_idxs[len(sent_data_type_list)]
            sent_example_list = []
            valid_ctkidx2sentidxs = [None] * len(self.ctk_list)
            valid_sentidx2ctkidxs = [None] * num_sents
            valid_sentidx2len = [None] * num_sents
            valid_sentidxs = []
            for _data_type in sent_data_type_list:
                _idx_dt = self.data_type_list.index(_data_type)
                for _sentidx in range(self.sent_part_idxs[_idx_dt], self.sent_part_idxs[_idx_dt+1]):
                    _meta = self._load_meta(_sentidx, _data_type)
                    _sent_ctkidxs, _sent_len = _meta[0], _meta[1]

                    if not self.is_valid_sent(_meta, _data_type):
                        if _data_type == "omcs" and self.use_invalid_omcs:
                            sent_data = load_sent_from_shard(
                                self.sent_index_offset_list[_sentidx], self.cached_dir, _data_type)
                            _sent_ctkidxs = [self.ctk2idx[_ctk] for _ctk in sent_data[2]]
                        else:
                            continue
                    valid_sentidx2ctkidxs[_sentidx] = _sent_ctkidxs
                    valid_sentidx2len[_sentidx] = _sent_len
                    valid_sentidxs.append(_sentidx)
                    for _ctkidx in _sent_ctkidxs:
                        if valid_ctkidx2sentidxs[_ctkidx] is None:
                            valid_ctkidx2sentidxs[_ctkidx] = set()
                        valid_ctkidx2sentidxs[_ctkidx].add(_sentidx)
            # clustering based on valid_cididx2sentidxs and valid_dsentidx2cididxs
            use_acc =  False  #  len(valid_sentidxs) > 1e6 and self.threshold_stop_ctk < 10
            random.shuffle(valid_sentidxs)
            used_sentidx_list = [False] * num_sents
            for _valid_sentidx in tqdm(valid_sentidxs):
                if used_sentidx_list[_valid_sentidx]:
                    continue

                _related_sentidxs = self._enrich_sent_wrt_limit(
                    _valid_sentidx, self.max_context_len,
                    valid_ctkidx2sentidxs, valid_sentidx2ctkidxs, valid_sentidx2len,
                    banned_sentidx_set=used_sentidx_list, use_acc=use_acc
                )
                _related_sentidxs.insert(0, _valid_sentidx)
                sent_example_list.append(_related_sentidxs)
                for _used_sentidx in _related_sentidxs:
                    used_sentidx_list[_used_sentidx] = True
                    for _ctkidx in valid_sentidx2ctkidxs[_used_sentidx]:
                        valid_ctkidx2sentidxs[_ctkidx].remove(_used_sentidx)

            save_pickle(sent_example_list, sent_example_path)

        # part 2:
        para_data_type_list = [_data_type for _data_type in self.data_type_list if _data_type in self.PARA_COPUS]
        para_example_list = []
        for _data_type in para_data_type_list:
            para_example_path = join(
                self.cached_dir, self.example_filename_template.format(_data_type, "para", "pkl"))
            # if exist
            if file_exists(para_example_path):
                print("\texample pickle for {} is found, loading.".format(_data_type))
                exs = load_pickle(para_example_path)
            else:
                print("\texample pickle for {} is not found, building.".format(_data_type))
                exs = []
                ex_lens = []
                print("\t\tbuilding buffer")
                buffer_sent_idx = []
                buffer_sent_len = []
                buffer_sent_valid = []
                _idx_dt = self.data_type_list.index(_data_type)
                for _sentidx in range(self.sent_part_idxs[_idx_dt], self.sent_part_idxs[_idx_dt + 1]):
                    _meta = self._load_meta(_sentidx, _data_type)
                    _sent_cididxs, _sent_len = _meta[0], _meta[1]
                    buffer_sent_idx.append(_sentidx)
                    buffer_sent_len.append(_sent_len)
                    buffer_sent_valid.append(self.is_valid_sent(_meta, _data_type))
                print("\t\tbuilding examples")
                buffer_ptr = 0
                while buffer_ptr < len(buffer_sent_idx):
                    # get the max end
                    max_end_ptr = buffer_ptr + 1
                    while True:
                        if max_end_ptr == len(buffer_sent_idx) or \
                                sum(buffer_sent_len[buffer_ptr: max_end_ptr+1]) > self.max_context_len:
                            break
                        max_end_ptr += 1

                    end_ptr = None
                    for _tmp_end_ptr in range(max_end_ptr, buffer_ptr, -1):
                        if np.mean(buffer_sent_valid[buffer_ptr: _tmp_end_ptr]) > 0.65:
                            end_ptr = _tmp_end_ptr
                            break
                    if end_ptr is None:  # current sent is invalid
                        buffer_ptr += 1
                    else:
                        exs.append(buffer_sent_idx[buffer_ptr: end_ptr])
                        ex_lens.append(sum(buffer_sent_len[buffer_ptr: end_ptr]))
                        buffer_ptr = end_ptr
                save_pickle(exs, para_example_path)
            # todo: combine examples w.r.t len
            # extend all list
            para_example_list.extend(exs)

        example_list = sent_example_list + para_example_list
        return example_list

    @staticmethod
    def _check_conflict(token_idxs_list, new_idxs):
        if len(new_idxs) == 0:
            return False
        return all(
            _idxs[-1] < new_idxs[0] or new_idxs[-1] < _idxs[0]
            for _idxs in token_idxs_list)

    @staticmethod
    def _get_unused_idxs_and_lens(seq_len, token_idxs_list):
        _used_set = set(_idx for _idxs in token_idxs_list for _idx in _idxs)
        assert len(_used_set) == 0 or max(_used_set) < seq_len

        tk_idx2len_dict = {}
        idx_buffer = []
        for _idx in range(seq_len + 1):  # for final one
            if _idx in _used_set or _idx == seq_len:
                if len(idx_buffer) > 0:
                    for _j, _i in enumerate(idx_buffer):
                        tk_idx2len_dict[_i] = len(idx_buffer) - _j
                    idx_buffer = []
            else:
                idx_buffer.append(_idx)
        return tk_idx2len_dict  # {id==>max_span_len_available}

    @staticmethod
    def _geometric_sampling(geo_p, max_len):
        res = max_len + 1
        while res > max_len:
            res = int(np.random.geometric(geo_p))
        return res

    @staticmethod
    def _new_idxs_for_idxs_list(all_idxs_list, cur_idxs_list):
        return set([_idx for _idxs in cur_idxs_list for _idx in _idxs]) - \
               set([_idx for _idxs in all_idxs_list for _idx in _idxs])