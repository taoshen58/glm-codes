from tqdm import tqdm
import networkx as nx
import numpy as np
from os.path import join
import argparse, time, json, pickle, os, collections, random
from configs import index_sent_cache_dir
from data_proc.concept_extractor import ConceptExtractor, NUM_CONCEPT_NET_REL, lemma_dict_path, \
    load_conceptnet_en_map_dump, conceptnet_dump_iter, conceptnet_en_path, REDUNDANT_RELATIONS, \
    load_list_from_file, conceptnet_rel2idx_out_path, triplet2sent_path, UNDIRECTED_RELATIONS
from utils.common import file_exists, dir_exists, load_json, save_json, parse_span_str, get_data_path_list, \
    get_statistics_for_num_list, load_pickle, save_pickle, save_jsonl_with_offset, save_list_to_file
from contextlib import contextmanager
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists


neighbor_cididxs_file_name = "cn_neighbor_cididxs.jsonl"
neighbor_cididxs_offset_file_name = "cn_neighbor_cididxs_offset.pkl"
stop_ctkidx_list_file_name = "cn_stop_ctkidx_list.pkl"
stop_ctk_list_file_name = "cn_stop_ctk_list.txt"

def load_conceptnet():
    # load lemma2tk_cid_list
    print("load all phrase and cid from conceptnet")
    s_t = time.time()
    lemma2ids = load_conceptnet_en_map_dump(lemma_dict_path)
    ctoken_set = set()
    cid_set = set()
    for _tk_cid_list in tqdm(lemma2ids.values(), total=len(lemma2ids)):
        for _tk, _cid in _tk_cid_list:
            ctoken_set.add(_tk)
            cid_set.add(_cid)
    ctk_list = list(sorted(ctoken_set))
    cid_list = list(sorted(cid_set))
    ctk2idx = dict((_tk, _i) for _i, _tk in enumerate(ctk_list))
    cid2idx = dict((_cid, _i) for _i, _cid in enumerate(cid_list))

    cididx2ctkidx = [None for _ in range(len(cid_list))]
    ctkidx2cididxs = [[] for _ in range(len(ctk_list))]
    for _tk_cid_list in tqdm(lemma2ids.values(), total=len(lemma2ids)):
        for _tk, _cid in _tk_cid_list:
            _cididx = cid2idx[_cid]
            _ctkidx = ctk2idx[_tk]
            cididx2ctkidx[_cididx] = _ctkidx
            ctkidx2cididxs[_ctkidx].append(_cididx)
    print("\tDone, using {}".format(time.time() - s_t))
    return ctk_list, cid_list, ctk2idx, cid2idx, cididx2ctkidx, ctkidx2cididxs


def load_conceptnet_graph(cid_list, cid2idx, load_graph=True,):  # build complete graph from conceptNet
    print("loading graph")
    s_t = time.time()
    rel_list = load_list_from_file(conceptnet_rel2idx_out_path)
    rel2idx = dict((_rel, _idx_rel) for _idx_rel, _rel in enumerate(rel_list))
    cg = nx.Graph()
    if not load_graph:
        print("\tDone with rel instead of graph, using {}".format(time.time() - s_t))
        return rel_list, rel2idx, None, None

    for _row in tqdm(conceptnet_dump_iter(conceptnet_en_path)):
        _id1, _id2 = _row[2], _row[3]
        _cididx1, _cididx2 = cid2idx[_id1], cid2idx[_id2]
        _rel = _row[1]
        _source = json.loads(_row[4])
        cg.add_edge(
            _cididx1, _cididx2,
            relation=rel2idx[_rel], weight=_source.get("weight") or 1.
        )
    cg = cg
    cididx2neighbor = []
    for _cididx in tqdm(range(len(cid_list)), desc="CG2LIST"):
        try:
            cididx2neighbor.append(cg[_cididx])
        except KeyError:
            cididx2neighbor.append(dict())
    print("\tDone, using {}".format(time.time() - s_t))
    return rel_list, rel2idx, cg, cididx2neighbor


def load_stop_ctkidx_list(cache_dir, stop_prop=1):
    # assert stop_prop in stop_ctk_list_proportion_list
    loaded_list = load_pickle(join(cache_dir, stop_ctkidx_list_file_name))
    need_num = int(1.* stop_prop / 1000 * len(loaded_list))
    needed_list = loaded_list[:need_num]
    return needed_list


def load_sent_index_offset(data_type, cache_dir):
    return load_pickle(join(cache_dir, data_type, "sent_index_list_offset.pkl"))


def load_sent_index(sent_index_info, cache_dir, data_type=None):
    if isinstance(sent_index_info, int):
        assert data_type is not None
        with open(join(cache_dir, data_type, "sent_index_list"), encoding="utf-8") as fp:
            fp.seek(sent_index_info)
            sent_index_info = json.loads(fp.readline())
            return sent_index_info
    else:
        return sent_index_info


def load_sent_from_shard(sent_index_info, cache_dir, data_type=None):
    sent_index_info = load_sent_index(sent_index_info, cache_dir, data_type)
    assert len(sent_index_info) == 5
    shard_path = join(cache_dir, "{}", "{}-shard-{}.jsonl").format(
        sent_index_info[0], sent_index_info[0], sent_index_info[3])
    file_offset = sent_index_info[4]
    with open(shard_path, encoding="utf-8") as rfp:
        rfp.seek(file_offset)
        data = json.loads(rfp.readline())
    return data


def load_neighbor_cididxs_offsets(cache_dir):
    return load_pickle(join(cache_dir, neighbor_cididxs_offset_file_name))


def load_neighbor_cididxs(ct_cididx, offsets, cache_dir):
    offset = offsets[ct_cididx]
    with open(join(cache_dir, neighbor_cididxs_file_name), encoding="utf-8") as fp:
        fp.seek(offset)
        return json.loads(fp.readline())


def get_data_type(idx_sent, part_idxs, data_type_list):
    flags = [1 if part_idxs[_i] <= idx_sent < part_idxs[_i + 1] else 0 for _i in range(len(data_type_list))]
    return data_type_list[np.argmax(flags)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type_list", type=str, default="omcs,arc")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--k_hop", type=int, default=3)
    parser.add_argument("--max_num_nodes", type=int, default=1024)
    parser.add_argument("--disable_stop_ctk", action="store_true")
    parser.add_argument("--disable_nb", action="store_true")
    args = parser.parse_args()

    data_type_list = args.data_type_list.split(",")
    num_workers = args.num_workers
    cache_dir = args.cache_dir or index_sent_cache_dir
    k_hop = args.k_hop
    max_num_nodes = args.max_num_nodes
    disable_stop_ctk = args.disable_stop_ctk
    disable_nb = args.disable_nb
    data_type_list = [_e for _e in ["gen", "omcs", "arc", "wikipedia"] if _e in data_type_list]
    ctk_list, cid_list, ctk2idx, cid2idx, cididx2ctkidx, ctkidx2cididxs = load_conceptnet()
    rel_list, rel2idx, cg, cididx2neighbor = load_conceptnet_graph(cid_list, cid2idx)

    part_idxs = [0, ]
    sent_index_offset_list = []
    for _data_type in data_type_list:
        _offset_list = load_sent_index_offset(_data_type, cache_dir)
        sent_index_offset_list.extend(_offset_list)
        part_idxs.append(len(sent_index_offset_list))

    # read all sent
    if disable_stop_ctk:
        print("disable_stop_ctk!!!!!")
    else:
        print("reading all sent to count ctkidx2freq")
        ctkidx2freq_path = join(cache_dir, "cn_ctkidx2freq.pkl")
        if file_exists(ctkidx2freq_path):
            print("\tfound file, loading")
            ctkidx2freq = load_pickle(ctkidx2freq_path)
        else:
            print("\tnot found file, building")
            def _processor_ctkidx2freq(_sent_index_offset_list, _with_sent_index=False):
                local_ctkidx2freq = [0 for _ in range(len(ctk_list))]

                if _with_sent_index:
                    _iterator = tqdm(_sent_index_offset_list)
                else:
                    _iterator = enumerate(tqdm(_sent_index_offset_list))

                for _idx_sent, _sent_index_offset in _iterator:
                    _data_type = get_data_type(_idx_sent, part_idxs, data_type_list)
                    if _data_type != "gen":
                        _sent_data = load_sent_from_shard(_sent_index_offset, cache_dir, _data_type)
                        _tk2spans = _sent_data[2]
                        for _tk in _tk2spans:
                            local_ctkidx2freq[ctk2idx[_tk]] += 1
                return local_ctkidx2freq
            if num_workers == 1:
                ctkidx2freq = _processor_ctkidx2freq(sent_index_offset_list)
            else:
                sent_index_offset_list_with_index = list((_idx, _e) for _idx, _e in enumerate(sent_index_offset_list))
                local_ctkidx2freq_list = multiprocessing_map(
                    _processor_ctkidx2freq, dict_args_list=[
                        {"_sent_index_offset_list": _d, "_with_sent_index": True}
                        for _d in split_to_lists(sent_index_offset_list_with_index, num_workers)
                    ], num_parallels=num_workers
                )
                ctkidx2freq = [sum(_ll[_ctkidx] for _ll in local_ctkidx2freq_list) for _ctkidx in range(len(ctk_list))]
            save_pickle(ctkidx2freq, ctkidx2freq_path)
        print("\tDone")

        # sorting
        print("Getting stop ctk")
        sorted_ctkidx_freq_pairs = sorted(
            [(_ctkidx, _freq) for _ctkidx, _freq in enumerate(ctkidx2freq) if _freq > 0],
            key=lambda _e: _e[1], reverse=True)
        sorted_ctkidx_list, _ = [list(_e) for _e in zip(*sorted_ctkidx_freq_pairs)]
        save_pickle(sorted_ctkidx_list, join(cache_dir, stop_ctkidx_list_file_name))
        save_list_to_file([ctk_list[_ctkidx] for _ctkidx in sorted_ctkidx_list],
                          join(cache_dir, stop_ctk_list_file_name))
        print("\tDone")

    # find
    def _processor(_cididx_list):
        _local_res_list = []
        for _ct_cididx in tqdm(_cididx_list):
            _node_explored = set([_ct_cididx])
            _node_save = [[_ct_cididx], ] + [[] for _ in range(k_hop)]
            _node_buffer = [(_ct_cididx, 0)]
            while len(_node_buffer) > 0:
                _node_cididx, _prev_depth = _node_buffer.pop(0)
                if _prev_depth == k_hop:
                    continue
                _cur_depth = _prev_depth + 1
                _neighbors = cididx2neighbor[_node_cididx]
                # shuffle keys
                _nb_cididxs = list(_neighbors.keys())
                random.shuffle(_nb_cididxs)
                for _nb_cididx in _nb_cididxs:
                    _attr = _neighbors[_nb_cididx]
                    if _nb_cididx in _node_explored:
                        continue
                    _node_explored.add(_nb_cididx)
                    _node_buffer.append((_nb_cididx, _cur_depth))
                    if rel_list[_attr["relation"]] not in REDUNDANT_RELATIONS:  # remove REDUNDANT_RELATIONS
                        _node_save[_cur_depth].append(_nb_cididx)
                        if sum(len(_e) for _e in _node_save) > max_num_nodes:
                            _node_buffer = []
                            break

            _local_res_list.append(_node_save)
        return _local_res_list

    if disable_nb:
        print("disable_nb!!!!!")
    else:
        print("Getting neighbors")
        proc_buffer = []
        wfp_nb = open(join(cache_dir, neighbor_cididxs_file_name), "w", encoding="utf-8")
        nb_offsets = []
        for _ctkidx in tqdm(range(len(cid_list)), total=len(cid_list)):
            proc_buffer.append(_ctkidx)
            if len(proc_buffer) == num_workers * 10000 or _ctkidx == (len(cid_list)-1):
                if num_workers == 1:
                    _res_list = _processor(proc_buffer)
                else:
                    _res_list = combine_from_lists(
                        multiprocessing_map(
                            _processor, dict_args_list=[
                                {"_cididx_list": _d} for _d in split_to_lists(proc_buffer, num_parallels=num_workers)
                            ], num_parallels=num_workers
                        ), ordered=True
                    )
                assert len(_res_list) == len(proc_buffer)
                for _elem in _res_list:
                    nb_offsets.append(wfp_nb.tell())
                    _dump_str = json.dumps(_elem) + os.linesep
                    wfp_nb.write(_dump_str)
                proc_buffer = []
        wfp_nb.close()
        save_pickle(nb_offsets, join(cache_dir, neighbor_cididxs_offset_file_name))
        print("\tDone")


if __name__ == '__main__':
    main()
