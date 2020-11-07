from utils.common import save_list_to_file, load_list_from_file, file_exists, save_json
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists

import json
from tqdm import tqdm
import networkx as nx
import csv
import spacy
import collections
from configs import conceptnet_dir
import os


END_PHRASES = ["or other thing", "other thing", "thing", "something", "anything", "somebody", "anybody"]
END_PHRASES_LISTS = list(
    sorted([_p.strip().split(" ") for _p in END_PHRASES], key=lambda _l: len(_l), reverse=True)
)
MIDDLE_PHRASE_LIST = set(["the", "a", "an"])


def conceptnet_dump_iter(path_to_conceptnet, parse_json=False):
    with open(path_to_conceptnet, encoding="utf-8") as fp:
        csv_reader = csv.reader(fp, dialect="excel-tab")
        for _idx_r, _row in enumerate(csv_reader):
            if parse_json:
                _row[-1] = json.loads(_row[-1])
            yield _row

# TO: remove some asterisk wildcards and remove "a/an/the" for the concept in conceptnet
def clean_phrase(_phrase):
    _phrase = _phrase.lower()
    _token_list = _phrase.split()
    # * remove the ends
    for _end_token_list in END_PHRASES_LISTS:
        if len(_token_list) > len(_end_token_list):
            if _token_list[-len(_end_token_list):] == _end_token_list:
                _token_list = _token_list[:-len(_end_token_list)]
    # * remove the middle items
    for _idx in range(len(_token_list)-1, -1, -1):
        if len(_token_list) == 1:
            break
        elif _token_list[_idx] in MIDDLE_PHRASE_LIST:
            _token_list.pop(_idx)

            assert len(_token_list) > 0
    return " ".join(_token_list)


def load_conceptnet_en_map_dump(path_to):
    with open(path_to, encoding="utf-8") as fp:
        return json.load(fp)

# =========
def build_conceptnet_en_map(en_path, num_parallels):
    # build a dict: lemma str 2 conceptnet-id
    def _process(_conceptnet_id_list):
        nlp = spacy.load("en", disable=["parser", "ner", "textcat"])
        new_lemma_conceptnetid = collections.defaultdict(list)
        for _concept in tqdm(_conceptnet_id_list):
            _proc_concept = _concept.split("/")[3]
            _proc_concept = " ".join(_proc_concept.split("_"))
            doc = nlp(_proc_concept)
            _proc_concept = " ".join([token.lemma_ for token in doc])
            new_lemma_conceptnetid[_proc_concept].append(_concept)
        return new_lemma_conceptnetid

    concept_set = set()
    for _row in conceptnet_dump_iter(en_path):
        for _concept in _row[2:4]:
            if _concept not in concept_set:
                concept_set.add(_concept)
    concept_list = list(concept_set)

    multi_dict = multiprocessing_map(
        func=_process,
        dict_args_list=[{"_conceptnet_id_list": _data}
                        for _data in split_to_lists(concept_list, num_parallels)],
        num_parallels=num_parallels
    )

    final_dict = combine_from_lists(multi_dict, elem_type="dict")

    for _lemma, _vals in final_dict.items():
        for _idx_v in range(len(_vals)):
            _proc_concept = _vals[_idx_v].split("/")[3]
            _proc_concept = " ".join(_proc_concept.split("_"))
            _vals[_idx_v] = [_proc_concept, _vals[_idx_v]]

    return final_dict


# =========
def clean_non_english_item(path_to_conceptnet, out_path):
    fp_w = open(out_path, "w", encoding="utf-8")  #
    csv_writer = csv.writer(fp_w, "excel-tab")

    num_all_triplet, num_tgt_all_triplet = 0, 0
    with open(path_to_conceptnet, encoding="utf-8") as fp:
        csv_reader = csv.reader(fp, dialect="excel-tab")
        for _idx_r, _row in tqdm(enumerate(csv_reader), total=34074917):
            num_all_triplet += 1
            not_en_flag = False
            for _item in [_row[2], _row[3]]:
                if _item.split("/")[2] != "en":
                    not_en_flag = True
                    break
            if not_en_flag:
                continue

            num_tgt_all_triplet += 1
            csv_writer.writerow(_row)

    fp_w.close()
    print("{}/{}".format(num_tgt_all_triplet, num_all_triplet))


def build_conceptnet_en_map_dump(conceptnet_en_path, num_parallels, outpath):
    final_dict = build_conceptnet_en_map(conceptnet_en_path, num_parallels=num_parallels)
    with open(outpath, "w", encoding="utf-8") as fp:
        json.dump(final_dict, fp)
    return True


def build_conceptnet_id2idx(conceptnet_en_path, out_path):
    conceptnet_id_set = set()
    for _row in tqdm(conceptnet_dump_iter(conceptnet_en_path)):
        for _concept in _row[2:4]:
            conceptnet_id_set.add(_concept)
    conceptnet_id_list = list(sorted(list(conceptnet_id_set)))
    save_list_to_file(conceptnet_id_list, out_path)


def build_conceptnet_rel2idx(conceptnet_en_path, out_path):
    conceptnet_rel_set = set()
    for _row in tqdm(conceptnet_dump_iter(conceptnet_en_path)):
        conceptnet_rel_set.add(_row[1])
    conceptnet_rel_list = list(sorted(list(conceptnet_rel_set)))
    save_list_to_file(conceptnet_rel_list, out_path)


def build_graph_using_networkx(conceptnet_en_path, conceptnet_id2idx_path):
    g = nx.Graph()
    for _row in conceptnet_dump_iter(conceptnet_en_path):
        g.add_edge(_row[2], _row[3])
    return g


def build_clean_lemma2tags(conceptnet_en_path, dump_path, num_parallels=20):
    # build a dict: clean_lemma str 2 list of tagger
    def _process(_conceptnet_id_list):
        nlp = spacy.load("en", disable=["parser", "ner", "textcat"])
        new_lemma_conceptnetid = collections.defaultdict(list)
        for _concept in tqdm(_conceptnet_id_list):
            _proc_concept = _concept.split("/")[3]
            _proc_concept = " ".join(_proc_concept.split("_"))
            doc = nlp(_proc_concept)
            _proc_concept = " ".join([token.lemma_ for token in doc])
            _clean_concept = clean_phrase(_proc_concept)
            if _clean_concept not in new_lemma_conceptnetid:
                _attr_list = [[token.tag_, ] for token in nlp(_clean_concept)]
                new_lemma_conceptnetid[_clean_concept] = _attr_list
        return new_lemma_conceptnetid

    concept_set = set()
    for _row in conceptnet_dump_iter(conceptnet_en_path):
        for _concept in _row[2:4]:
            if _concept not in concept_set:
                concept_set.add(_concept)
    concept_list = list(concept_set)

    multi_dict = multiprocessing_map(
        func=_process,
        dict_args_list=[{"_conceptnet_id_list": _data}
                        for _data in split_to_lists(concept_list, num_parallels)],
        num_parallels=num_parallels
    )

    final_dict = {}
    for _dict in multi_dict:
        final_dict.update(_dict)

    save_json(final_dict, dump_path)


def conceptnet_preprocess_pipeline(_conceptnet_path, num_workers=20, check_exist=True):
    _conceptnet_en_path = _conceptnet_path + "_en.csv"
    _lemma_dict_out_path = _conceptnet_en_path + "_lemma_dict.json"  # lemma to a list of []
    _conceptnet_id2idx_out_path = _conceptnet_en_path + "_id2idx.txt"
    _conceptnet_rel2idx_out_path = _conceptnet_en_path + "_rel2idx.txt"
    _clean_lemma2tags_out_path = _conceptnet_en_path + "_clean_lemma2tags.json"
    _path_list = [
        _conceptnet_en_path, _lemma_dict_out_path, _conceptnet_id2idx_out_path,
        _conceptnet_rel2idx_out_path, _clean_lemma2tags_out_path]

    if not (check_exist and all(file_exists(_path) for _path in _path_list)):
        if not file_exists(_conceptnet_en_path):
            clean_non_english_item(_conceptnet_path, _conceptnet_en_path)
        build_conceptnet_en_map_dump(_conceptnet_en_path, num_workers, _lemma_dict_out_path)
        build_conceptnet_id2idx(_conceptnet_en_path, _conceptnet_id2idx_out_path)
        build_conceptnet_rel2idx(_conceptnet_en_path, _conceptnet_rel2idx_out_path)
        build_clean_lemma2tags(_conceptnet_en_path, _clean_lemma2tags_out_path, num_parallels=num_workers)
    return _path_list


if __name__ == '__main__':
    conceptnet_path = os.path.join(conceptnet_dir, "assertions.csv")
    conceptnet_preprocess_pipeline(conceptnet_path, num_workers=20, check_exist=True)


