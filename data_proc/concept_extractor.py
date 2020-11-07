import networkx as nx
import elasticsearch
import collections
import os
import json
import time
import spacy
from tqdm import tqdm
from copy import deepcopy
import scipy.sparse as sparse
import numpy as np
from fuzzywuzzy import fuzz
import os

from conceptnet.pipeline import load_conceptnet_en_map_dump, conceptnet_dump_iter, clean_phrase
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists
from utils.common import load_list_from_file
from configs import conceptnet_dir
conceptnet_en_path = os.path.join(conceptnet_dir, "assertions.csv_en.csv")
triplet2template_path = os.path.join(conceptnet_dir, "template.json")
triplet2sent_path = os.path.join(conceptnet_dir, "triplet2sent.txt")
lemma_dict_path = conceptnet_en_path + "_lemma_dict.json"
conceptnet_id2idx_path = conceptnet_en_path + "_id2idx.txt"
conceptnet_rel2idx_out_path = conceptnet_en_path + "_rel2idx.txt"
clean_lemma2tags_out_path = conceptnet_en_path + "_clean_lemma2tags.json"


# # Stop words
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# print(stop_words)
import string
PUNCTUATIONS = set(string.punctuation)

REDUNDANT_RELATIONS = set([
    "/r/Antonym", "/r/DerivedFrom", "/r/EtymologicallyDerivedFrom", "/r/EtymologicallyRelatedTo",
    "/r/FormOf", "/r/Synonym", "/r/dbpedia/capital",
    "/r/dbpedia/field", "/r/dbpedia/genre", "/r/dbpedia/genus", "/r/dbpedia/influencedBy", "/r/dbpedia/knownFor",
    "/r/dbpedia/language", "/r/dbpedia/leader", "/r/dbpedia/occupation", "/r/dbpedia/product",

    # Somehow Useful
    # "/r/DistinctFrom",  # *
    # "/r/Entails",  # *
    # "/r/HasContext",  # *
    # "/r/MannerOf",  # *
    # "/r/SimilarTo",  # *

    # Too General
    # "/r/RelatedTo",
])

UNDIRECTED_RELATIONS = set([
    "/r/Antonym", "/r/DistinctFrom", "/r/EtymologicallyRelatedTo", "LocatedNear",
    "/r/RelatedTo", "/r/SimilarTo", "/r/Synonym",
])

STOP_WORDS = set([
    'a', 'an', 'the',
    'am', 'is', 'are', 'be', 'been', 'being',
])


NUM_CONCEPT_NET_REL = 47


class PhraseTree(object):
    def __init__(self, phrase_list):
        # build the tree
        self._phrase_tree = self._build_phrase_tree(phrase_list)

    def _build_phrase_tree(self, phrase_list):
        phrase_tree = {"type": "root", "children": {}}
        for _phrase in phrase_list:
            _wd_list = _phrase.split(" ")
            _cur_node = phrase_tree
            for _idx_w, _wd in enumerate(_wd_list):
                if _wd not in _cur_node["children"]:
                    _cur_node["children"][_wd] = {"type": "non-phrase", "children": {}}
                if _idx_w == len(_wd_list) - 1:  # this is the last word is the phrase
                    _cur_node["children"][_wd]["type"] = "phrase"
                _cur_node = _cur_node["children"][_wd]
        return phrase_tree

    def _res_phrase_scoring(self, _res_range):
        return (-len(_res_range) / (_res_range[-1][1] - _res_range[0][1] + 1), -len(_res_range))

    def search_phrase(self, lemma_list, max_gaps):
        if len(lemma_list) == 0:
            return []
        if isinstance(lemma_list[0], dict):
            lemma_list = [_tk["lemma"] for _tk in lemma_list]

        lemma_lists = [list(zip(lemma_list[_idx:], range(_idx, len(lemma_list)))) for _idx in range(len(lemma_list))]

        init_len = len(lemma_lists)
        lemma_lists_buffer = lemma_lists
        cur_node_buffer = [self._phrase_tree for _ in range(init_len)]
        tmp_res_buffer = [[] for _ in range(init_len)]

        res_list = [[] for _ in range(init_len)]

        while len(lemma_lists_buffer) > 0:
            new_lemma_lists_buffer = []
            new_cur_node_buffer = []
            new_tmp_res_buffer = []
            for _idx_bf, _lemma_list in enumerate(lemma_lists_buffer):

                for _idx_tk, (_lemma_text, _lemma_idx) in enumerate(_lemma_list):
                    if _idx_tk > max_gaps:  # looseness
                        break

                    if _lemma_text in PUNCTUATIONS:  # if encounter punkt, break the search
                        break

                    _init_flag = (len(tmp_res_buffer[_idx_bf]) == 0)

                    if _lemma_text in cur_node_buffer[_idx_bf]["children"]:
                        _new_node = cur_node_buffer[_idx_bf]["children"][_lemma_text]
                        _new_lemma_list = _lemma_list[(_idx_tk + 1):]
                        _new_tmp_res = tmp_res_buffer[_idx_bf] + [[_lemma_text, _lemma_idx, "CONCEPT"], ]

                        if _new_node["type"] == "phrase":
                            res_list[_new_tmp_res[0][1]].append(deepcopy(_new_tmp_res))

                        if len(_new_lemma_list) > 0:
                            new_lemma_lists_buffer.append(_new_lemma_list)
                            new_cur_node_buffer.append(_new_node)
                            new_tmp_res_buffer.append(_new_tmp_res)

                    if _init_flag:
                        break
            lemma_lists_buffer = new_lemma_lists_buffer
            cur_node_buffer = new_cur_node_buffer
            tmp_res_buffer = new_tmp_res_buffer

        # add filter for res_list
        for _idx_elems, _elems in enumerate(res_list):
            _new_elems = []
            _phrases = []
            for _e in _elems:
                if len(_e) == 1:
                    _new_elems.append(_e)
                else:
                    _phrases.append(_e)
            if len(_phrases) > 0:
                _new_elems.append(list(sorted(_phrases, key=self._res_phrase_scoring))[0])
            # in-place replacement
            res_list[_idx_elems] = _new_elems

        # return the final results
        return res_list


class ConceptExtractor(object):
    TARGET_ENT_TYPES = [
        "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
    ]

    def __init__(self):
        # build lemma list
        s_t = time.time()
        print("building lemma list")
        lemma2ids = load_conceptnet_en_map_dump(lemma_dict_path)  # ***
        # # clean the lemma
        new_lemma2ids = collections.defaultdict(list)
        for _lemma, _ids in lemma2ids.items():
            _clemma = clean_phrase(_lemma)
            new_lemma2ids[_clemma].extend(_ids)
        _lemma2ids = new_lemma2ids

        print("\tTime is", time.time() - s_t)

        # build lemma aux variable for acceleration
        s_t = time.time()
        print("building lemma aux variable for acceleration")
        lemma_all_list = list(_lemma2ids.keys())
        print("\tTime is", time.time() - s_t)

        s_t = time.time()
        print("building phrase tree object")
        pt = PhraseTree(lemma_all_list)
        print("\tTime is", time.time() - s_t)

        s_t = time.time()
        print("building spacy pipeline")
        nlp = spacy.load("en", disable=["parser", "textcat"])  # "ner",
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp_small = spacy.load("en", disable=["parser", "ner", "textcat"])
        print("\tTime is", time.time() - s_t)

        # === register member variables ===
        self._lemma2ids = _lemma2ids

        self._nlp = nlp
        self._nlp_small = nlp_small
        self._pt = pt

        # Graph related
        self._cg = None
        self._cg_id = None
        self._node_list = None
        self._node2idx = None
        self._neighbor_mat = None
        self._rel_list = None

    def extract_phrase_from_text(self, passage, max_gap=2, remove_stop_words=True, out=None):
        # load from member variables
        nlp = self._nlp
        nlp_small = self._nlp_small
        pt = self._pt
        TARGET_ENT_TYPES = self.TARGET_ENT_TYPES  # only retrieve the entity in this type set
        # =======
        _doc = nlp(passage)
        _phrase2char_span = {}
        for _s in _doc.sents:  # needed attributes from spacy
            _sent = []
            for _tk in _s:
                _tk_dict = {
                    "ent_type": _tk.ent_type_, "ent_iob": _tk.ent_iob_, "lemma": _tk.lemma_.lower(),
                    "char_start": _tk.idx, "char_end": _tk.idx + len(_tk.text), "token_idx": _tk.i
                }
                _sent.append(_tk_dict)
            if len(_sent) == 0:
                continue

            # re-do lemma for entities to match the lemma in concept net
            # WHY: because spacy will omit the lemma for capital named entity
            for _tk in _sent:
                if _tk["ent_iob"] != "O" and _tk["ent_type"] in TARGET_ENT_TYPES:  # condition
                    _lemma_doc = nlp_small(_tk["lemma"])
                    _new_lemma = [_t.lemma_ for _t in _lemma_doc]
                    _tk["lemma"] = "".join(_new_lemma)

            # mark the entity in some types as concept
            _spans = []  #
            _cur_span = []
            for _idx_tk, _tk in enumerate(_sent):
                if _tk["ent_iob"] == "B" and _tk["ent_type"] in TARGET_ENT_TYPES:
                    if len(_cur_span) > 0:
                        _spans.append(_cur_span)
                        _cur_span = []
                    _cur_span.append([_tk["lemma"], _idx_tk, _tk["ent_type"]])
                elif _tk["ent_iob"] == "I" and _tk["ent_type"] in TARGET_ENT_TYPES:
                    _cur_span.append([_tk["lemma"], _idx_tk, _tk["ent_type"]])

                if _tk["ent_iob"] == "O" or _idx_tk == len(_sent) - 1:
                    if len(_cur_span) > 0:
                        _spans.append(_cur_span)
                        _cur_span = []

            # # clean det. WHY: SpaCy always consider article as a part of entity
            _new_spans = []
            for _span in _spans:
                _new_span = [_elem for _elem in _span if _elem[0] not in ["a", "an", "the"]]
                if len(_new_span) > 0:
                    _new_spans.append(_new_span)
            _spans = _new_spans

            # used token idx
            _used_idx_set = set()
            for _span in _spans:
                _used_idx_set.update(range(_span[0][1], _span[-1][1] + 1))

            # search in phrase tree
            _search_spans_list = pt.search_phrase(_sent, max_gaps=max_gap)  # a list (len=sent_len) of lemma_list

            # filter search span:
            #   1) not oppsite to _spans !!! temporarily REMOVED
            # _search_spans_list = [[_span for _span in _sps if all(_elem[1] not in _used_idx_set for _elem in _span)]
            #                       for _sps in _search_spans_list]
            #   2) remove the result whose is a single word and is also stop word
            for _idx_tk, _tk in enumerate(_s):
                if _tk.text.lower() in STOP_WORDS or (_tk.is_stop and remove_stop_words):  # text.lower() in STOP_WORDS:
                    _new_list = []
                    for _lst in _search_spans_list[_idx_tk]:
                        if len(_lst) > 1:
                            _new_list.append(_lst)
                    _search_spans_list[_idx_tk] = _new_list

            # insert entity span backed to _search_spans_list
            for _span in _spans:
                _search_spans_list[_span[0][1]].append(_span)

            # list all phrase
            _phrase_dict = collections.defaultdict(
                lambda: {"idx_set": set(), "span_list": set(), "neighbor_phrase_list": [], "type": None})
            for _search_spans in _search_spans_list:
                for _span in _search_spans:
                    _phrase = " ".join([_sent[_elem[1]]["lemma"] for _elem in _span])
                    _idx_set = set(range(_span[0][1], _span[-1][1] + 1))
                    _span_list = [_elem[1] for _elem in _span]
                    _phrase_dict[_phrase]["idx_set"].update(_idx_set)  # not used
                    _phrase_dict[_phrase]["span_list"].add(tuple(_span_list))
                    # type
                    if _phrase_dict[_phrase]["type"] is None or _phrase_dict[_phrase]["type"] == "CONCEPT":
                        _phrase_dict[_phrase]["type"] = _span[0][2]

            for _phrase, _attr in _phrase_dict.items():
                _type = _attr["type"]
                _span_list = []
                for _tk_span in list(_attr["span_list"]):
                    _span_list.append([_sent[_tk_span[0]]["char_start"], _sent[_tk_span[-1]]["char_end"]])

                if _phrase not in _phrase2char_span:
                    _phrase2char_span[_phrase] = [[], None]
                _phrase2char_span[_phrase][0].extend(_span_list)
                _phrase2char_span[_phrase][1] = _type

        if out is not None:
            out["doc_len"] = len(_doc)
            out["spacy_doc"] = _doc
        return _phrase2char_span

    # Improve Efficiency and combine related subgraph, v/s the old one
    def load_conceptnet_graph(self, max_hop=1):
        if self._cg is not None:
            return True

        s_t = time.time()
        print("load relation list ...")
        rel_list = load_list_from_file(conceptnet_rel2idx_out_path)
        rel2idx_dict = dict((_rel, _idx_rel) for _idx_rel, _rel in enumerate(rel_list))
        print("\tTime is", time.time() - s_t)
        # build lemma (phrase) conection graph
        s_t = time.time()
        print("building graph...")
        _id2lemma_tk = dict((_id, (_l, _tk)) for _l, _ids in self._lemma2ids.items() for _tk, _id in _ids)
        cg = nx.Graph()
        cg_id = nx.Graph()
        for _row in conceptnet_dump_iter(conceptnet_en_path):
            _id1, _id2 = _row[2], _row[3]
            _rel = _row[1]
            if _rel in REDUNDANT_RELATIONS:
                continue
            _rel_idx = rel2idx_dict[_rel]
            # part 1: concept_id graph
            if _id1 != _id2:
                cg_id.add_edge(_id1, _id2, weight=_rel_idx)
                # part 2: lemma graph
                _l1, _l2 = _id2lemma_tk[_id1][0], _id2lemma_tk[_id2][0]
                _l1, _l2 = clean_phrase(_l1), clean_phrase(_l2)
                if _l1 != _l2:
                    # format the edge
                    try:
                        weight_old = cg[_l1][_l2]["weight"]
                    except KeyError:
                        weight_old = 0
                    weight_bin = bin(weight_old)[2:]
                    weight_bin = ["0"] * (len(rel_list) - len(weight_bin)) + list(weight_bin)
                    weight_bin[_rel_idx] = "1"
                    weight_new = int("".join(weight_bin), 2)
                    cg.add_edge(_l1, _l2, weight=weight_new)

        # # adjacency matrix
        # print("\tbuilding adjacency matrix...")
        # adj_mat = nx.adjacency_matrix(cg)
        node_list = list(cg.nodes())
        node2idx = dict((_node, _i) for _i, _node in enumerate(node_list))
        print("\tcalc neighbor_mat...")
        adj_mat = (nx.adjacency_matrix(cg) + sparse.eye(len(node_list), dtype="int64")).astype("bool_")

        neighbor_mat = adj_mat
        tmp_max_hop = max_hop - 1
        while tmp_max_hop > 0:
            neighbor_mat = neighbor_mat.dot(adj_mat)
            tmp_max_hop -= 1
        # neighbor_mat = neighbor_mat.dot(adj_mat)
        # print("\ttransformat neighbor_mat to set list...")  # this is too space-consuming
        # neighbor_set_list = [set(neighbor_mat[_i].nonzero()[1]) for _i in range(len(node_list))]
        print("\tTime is", time.time() - s_t)

        self._cg = cg
        self._cg_id = cg_id
        self._node_list = node_list
        self._node2idx = node2idx
        self._neighbor_mat = neighbor_mat
        self._rel_list = rel_list

    def extract_subgraph_in_phrase2char_span(self, phrase2char_span, raw_text):
        cg = self._cg
        cg_id = self._cg_id
        node_list = self._node_list
        node2idx = self._node2idx
        neighbor_mat = self._neighbor_mat

        assert node_list is not None

        # produce phrase2XXXcache
        phrase2tk_id_score_list = dict()
        for _phrase in phrase2char_span:
            texts = []
            for _span in phrase2char_span[_phrase][0]:
                texts.append(raw_text[_span[0]:_span[-1]].lower())

            tk_id_score_list = []
            for _tk, _id in self._lemma2ids[_phrase]:
                _tk_lower = _tk.lower()
                # scoring for _tk
                _score = sum([fuzz.ratio(_tk_lower, _text)/100 for _text in texts]) / len(texts)
                tk_id_score_list.append([_tk, _id, _score])

            # ranking for
            tk_id_score_list = list(sorted(tk_id_score_list, key=lambda e: e[-1], reverse=True))
            phrase2tk_id_score_list[_phrase] = tk_id_score_list

        cache_id2cg_id_sub = dict()
        edge_dict = dict()
        for _phrase in phrase2char_span:
            # _spans = phrase2char_span[_phrase][0]
            # _type = phrase2char_span[_phrase][1]

            # other phrase set (i.e., except for itself):
            _other_phrase_list = [_p for _p in phrase2char_span if _phrase != _p]
            if len(_other_phrase_list) == 0:
                continue
            # try on the graph
            try:
                # # Method
                _cur_node_idx = node2idx[_phrase]
                _neighbors_in_sent = []
                _rels_in_sent = []
                _sub_cg = None
                for _p in _other_phrase_list:
                    try:
                        _other_node_idx = node2idx[_p]
                        if neighbor_mat[_cur_node_idx, _other_node_idx]:
                            _neighbors_in_sent.append(_p)
                            if _sub_cg is None:  # cold boot
                                _sub_cg = cg[_phrase]
                            _all_rels = [int(_i) for _i in list(bin(_sub_cg[_p]["weight"])[2:])]
                            _all_rels = [0] * (len(self._rel_list) - len(_all_rels)) + _all_rels
                            _all_rels_sum = sum(_all_rels)
                            assert _all_rels_sum > 0
                            if sum(_all_rels) == 1:
                                _rel = int(np.argmax(_all_rels))
                            else:  # _all_rels_sum > 1
                                _pair_list = []
                                tk_id_score_list1 = phrase2tk_id_score_list[_phrase]
                                tk_id_score_list2 = phrase2tk_id_score_list[_p]
                                for _e1 in tk_id_score_list1:
                                    for _e2 in tk_id_score_list2:
                                        _score = _e1[-1] * _e2[-1]
                                        _pair_list.append([_e1, _e2, _score])
                                _pair_list = list(sorted(_pair_list, key=lambda p: p[-1], reverse=True))
                                _cur_rel = None
                                for _e1, _e2, _ in _pair_list:
                                    (_tk1, _id1, _sc1), (_tk2, _id2, _sc2) = _e1, _e2
                                    try:
                                        if _id1 in cache_id2cg_id_sub:
                                            _cur_rel = cache_id2cg_id_sub[_id1][_id2]["weight"]
                                        elif _id2 in cache_id2cg_id_sub:
                                            _cur_rel = cache_id2cg_id_sub[_id2][_id1]["weight"]
                                        else:
                                            cache_id2cg_id_sub[_id1] = cg_id[_id1]
                                            _cur_rel = cache_id2cg_id_sub[_id1][_id2]["weight"]
                                    except KeyError:
                                        pass
                                    if _cur_rel is not None:
                                        break
                                assert _cur_rel is not None
                                _rel = _cur_rel
                            _rels_in_sent.append(_rel)
                    except KeyError:
                        pass
                if len(_neighbors_in_sent) > 0:
                    assert len(_neighbors_in_sent) == len(_rels_in_sent)
                    for _v, _r in zip(_neighbors_in_sent, _rels_in_sent):
                        edge_dict[tuple(sorted([_phrase, _v]))] = _r
            except KeyError:
                pass
        edge_list = [list(_e) + [_r, ] for _e, _r in edge_dict.items()]
        return edge_list


if __name__ == '__main__':
    ce = ConceptExtractor()
    ce.load_conceptnet_graph(max_hop=1)

    raw_text = "Large international companies are involved in bauxite, iron ore, diamond, and gold mining operations."
    phrase2char_span = ce.extract_phrase_from_text(raw_text)
    edges = ce.extract_subgraph_in_phrase2char_span(phrase2char_span, raw_text)
    print(phrase2char_span, edges)

