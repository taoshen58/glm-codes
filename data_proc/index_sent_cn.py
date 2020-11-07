from tqdm import tqdm
import networkx as nx
import numpy as np
from os.path import join
import argparse, time, json, os, collections
from configs import index_sent_cache_dir
from data_proc.concept_extractor import ConceptExtractor, NUM_CONCEPT_NET_REL, lemma_dict_path, \
    load_conceptnet_en_map_dump, conceptnet_dump_iter, conceptnet_en_path, REDUNDANT_RELATIONS, \
    load_list_from_file, conceptnet_rel2idx_out_path, triplet2sent_path, UNDIRECTED_RELATIONS
from utils.common import file_exists, dir_exists, load_json, save_json, parse_span_str, get_data_path_list, \
    get_statistics_for_num_list, load_pickle, save_pickle, save_jsonl_with_offset
from contextlib import contextmanager
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists


def reindex_sents_from_shards(data_type, cached_dir):
    data_dir = join(cached_dir, data_type)
    shard_path_list = get_data_path_list(data_dir, ".jsonl")
    shard_idx_list = [int(_p.split("-")[-1].split(".")[0]) for _p in shard_path_list]
    shard_path_list = sorted(zip(shard_path_list, shard_idx_list), key=lambda _e: _e[1])
    shard_path_list, shard_idx_list = list(zip(*shard_path_list))  # sorted

    # begin to read
    sent_index_list = []
    for _shard_path, _shard_idx in tqdm(zip(shard_path_list, shard_idx_list), total=len(shard_path_list)):
        with open(_shard_path, encoding="utf-8") as rfp:
            _tell_pos = rfp.tell()
            _idx_l = 0
            _line = rfp.readline()
            while _line:
                _sent_data = json.loads(_line)
                _sent_id = _sent_data[0]
                _tk2spans = _sent_data[2]

                # [data_type, passage_idx, sent_idx, shard_idx, file_offset] --> def load_sent
                _basic_info = list(_sent_id.split("-"))
                _basic_info[1], _basic_info[2] = int(_basic_info[1]), int(_basic_info[2])
                _basic_info.append(_shard_idx)
                _basic_info.append(_tell_pos)
                sent_index_list.append(_basic_info)

                _tell_pos = rfp.tell()
                _idx_l += 1
                _line = rfp.readline()
    return sent_index_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    cached_dir = args.cached_dir or index_sent_cache_dir
    num_parallels = args.num_workers
    data_type = args.data_type
    assert data_type in ["gen", "omcs", "arc", "wikipedia"]

    sent_index_list = reindex_sents_from_shards(data_type, cached_dir)

    offsets = save_jsonl_with_offset(
        sent_index_list, path=join(cached_dir, data_type, "sent_index_list")
    )
    save_pickle(offsets, path=join(cached_dir, data_type, "sent_index_list_offset.pkl"))

if __name__ == '__main__':
    main()





