import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate, _use_shared_memory
import numpy as np
import sys
import os
from os.path import join
from tqdm import tqdm
import json
import math
import random
from copy import deepcopy
import networkx as nx
import collections
import argparse
import regex as re
from fuzzywuzzy import fuzz

from utils.common import file_exists, dir_exists, load_json, save_json, parse_span_str
from utils.mutli_proc import multiprocessing_map, combine_from_lists, split_to_lists
from data_proc.concept_extractor import ConceptExtractor
import spacy
import math
import nltk
from configs import USER_HOME, omcs_dir, arc_dir, wikipedia_dir, openbookqa_dir, bookcorpus_dir, index_sent_cache_dir


from utils.datasets.raw_text import OmcsRawTextLoader, ArcRawTextLoader, OpenbookqaRawTextLoader, WikipediaRawTextLoader, BookcorpusRawTextLoader
processor_dict = {
    "omcs": (OmcsRawTextLoader, omcs_dir),
    "arc": (ArcRawTextLoader, arc_dir),
    "wikipedia": (WikipediaRawTextLoader, wikipedia_dir),
    "openbookqa": (OpenbookqaRawTextLoader, openbookqa_dir),
    "bookcorpus": (BookcorpusRawTextLoader, bookcorpus_dir)
}

class IndexSent(object):
    def __init__(self, data_type, data_iter, cached_dir):

        self._data_type = data_type
        self._data_iter = data_iter
        self._cached_dir = cached_dir

        # make dir and path
        if not dir_exists(self._cached_dir):
            os.makedirs(self._cached_dir)
        self._dump_dir = join(cached_dir, data_type)
        if not dir_exists(self._dump_dir):
            os.makedirs(self._dump_dir)
        self._num_per_shard = 500000  # this corresponds to passage index (i.e., line in the data iterator)
        self._shard_path_template = join(self._dump_dir, self._data_type + "-shard-{}.jsonl")

        # prepare
        self._concept_extractor = ConceptExtractor()  # concept extractor at lemma level
        self._lemma2ids = self._concept_extractor._lemma2ids

    def process_corpus(self, num_parallels=28, buffer_size=1000):
        def buffer_processor(_sent_buffer):
            if num_parallels == 1:
                _res_list = self.processor_sent_list(_sent_buffer)
            else:
                _res_lists = multiprocessing_map(
                    self.processor_sent_list, dict_args_list=[
                        {"sent_list": _d} for _d in split_to_lists(_sent_buffer, num_parallels)
                    ], num_parallels=num_parallels
                )
                _res_list = combine_from_lists(_res_lists, elem_type="union", ordered=True)
            # post process at main process
            _final_res = []
            for cid2sp, ent2sp in _res_list:
                _final_res.append([cid2sp, ent2sp])
            return _final_res

        real_buffer_size = num_parallels * buffer_size

        idx_buffer, sent_buffer = [], []

        num_shard = 0
        num_written = 0

        writer_fp = open(self._shard_path_template.format(num_shard), "w", encoding="utf-8")

        for _idx_p, _passage in enumerate(self._data_iter):
            # split to sentence
            if self._data_type in ["omcs", "arc"]:
                _sent_list = [_passage[0]]
            else:
                print("!!!!!!!!!!!!!!!! BECAUSE data_type not in omcs and arc, so invoke sent_tokenize !!!!!")
                _sent_list = nltk.sent_tokenize(_passage[0])
            for _idx_s, _sent in enumerate(_sent_list):
                _id = "{}-{}-{}".format(self._data_type, _idx_p, _idx_s)
                sent_buffer.append(_sent)
                idx_buffer.append(_id)

                if len(sent_buffer) >= real_buffer_size:
                    # start to process !!!! CODE MUST BE SYNC with below
                    buffer_res = buffer_processor(sent_buffer)
                    assert len(buffer_res) == len(sent_buffer)
                    # save to file
                    for _id, _s, _olist in zip(idx_buffer, sent_buffer, buffer_res):
                        _to_dump = [_id, _s] + _olist
                        writer_fp.write(json.dumps(_to_dump))
                        writer_fp.write(os.linesep)
                        num_written += 1
                        if num_written % self._num_per_shard == 0:
                            writer_fp.close()
                            num_shard += 1
                            writer_fp = open(self._shard_path_template.format(num_shard), "w", encoding="utf-8")
                    idx_buffer, sent_buffer = [], []
        if len(sent_buffer) > 0:
            # start to process !!!! CODE MUST BE SYNC with above
            buffer_res = buffer_processor(sent_buffer)
            assert len(buffer_res) == len(sent_buffer)
            # save to file
            for _id, _s, _olist in zip(idx_buffer, sent_buffer, buffer_res):
                _to_dump = [_id, _s] + _olist
                writer_fp.write(json.dumps(_to_dump))
                writer_fp.write(os.linesep)
                num_written += 1
                if num_written % self._num_per_shard == 0:
                    writer_fp.close()
                    num_shard += 1
                    writer_fp = open(self._shard_path_template.format(num_shard), "w", encoding="utf-8")
            idx_buffer, sent_buffer = [], []


    def processor_sent_list(self, sent_list):
        res_list = []
        for _sent in tqdm(sent_list):
            ctk2span_dict, ent2span_dict = self.process_sent(_sent)
            res_list.append((ctk2span_dict, ent2span_dict))
        return res_list

    def process_sent(self, sent_text):
        proc_out = {}
        phrase2char_span = self._concept_extractor.extract_phrase_from_text(
            sent_text, max_gap=1, remove_stop_words=True, out=proc_out)
        # spacy_doc = proc_out["spacy_doc"]

        # begin match to token
        # cid2span_dict = collections.defaultdict(list)
        ctk2span_dict = collections.defaultdict(list)
        ent2span_dict = {}
        for _phrase, _char_spans in phrase2char_span.items():
            if _char_spans[1] != "CONCEPT":
                ent2span_dict[_phrase] = _char_spans[0]

            try:
                _tk_cid_list = self._lemma2ids[_phrase]
            except KeyError:
                continue
            if len(_tk_cid_list) == 0:
                continue

            for _idx_cs, _char_span in enumerate(_char_spans[0]):
                _raw_text = sent_text[_char_span[0]: _char_span[1]]
                # match by edit distance ratio
                _scores = [round(fuzz.ratio(_raw_text, _tk)) for _tk, _cid in _tk_cid_list]
                _max_score = max(_scores)

                _ctk_set = set()
                for _score, (_tk, _cid) in zip(_scores, _tk_cid_list):
                    if _score == _max_score and _score > 40:
                        if _tk not in _ctk_set:
                            _ctk_set.add(_tk)
                            ctk2span_dict[_tk].append(_char_span)
                        # cid2span_dict[_cid].append(_char_span)
        # print(ctk2span_dict)
        return ctk2span_dict, ent2span_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    data_type = args.data_type
    num_parallels = args.num_workers
    cache_dir = args.cache_dir or index_sent_cache_dir

    raw_data_class, data_dir_path = processor_dict[data_type]
    raw_data_loader = raw_data_class(data_type, data_dir_path)

    index_sent = IndexSent(
        data_type,
        raw_data_loader.passage_iter(),
        cache_dir
    )
    index_sent.process_corpus(num_parallels=num_parallels, buffer_size=5000)

if __name__ == '__main__':
    main()
    # data_type = "omcs"
    # raw_data_class, data_dir_path = processor_dict[data_type]
    # raw_data_loader = raw_data_class(data_type, data_dir_path)
    #
    # index_sent = IndexSent(
    #     data_type,
    #     raw_data_loader.passage_iter(),
    #     USER_HOME + "/PyData/commonsense/index_sent"
    # )
    # index_sent.process_corpus(num_parallels=10, buffer_size=5000)