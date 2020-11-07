import json
import pickle
import os
import spacy
import copy
from utils.mutli_proc import multiprocessing_map, split_to_lists, combine_from_lists
from utils.common import get_data_path_list, is_lower, is_capital, is_word
import nltk


class RawTextLoader(object):
    def __init__(self, data_type, data_dir):
        self._data_type = data_type
        self._data_dir = data_dir
        self._data_path_list = []  # list of tuple (dataset)

        self._spacy_nlp = None

    def _load_spacy_nlp(self):
        nlp = spacy.load("en", disable=["parser", "ner", "textcat"])  # "tagger",
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        self._spacy_nlp = nlp

    def passage_iter(self):
        """

        :return: passage str and metadata {"path":, "passage_idx", "char_idx?", }
        """
        for _data_path, _data_name in self._data_path_list:
            with open(_data_path, encoding="utf-8") as fp:
                _tell_pos = fp.tell()
                _idx_l = 0
                _line = fp.readline()
                while _line:
                    _meta_data = {
                        "data_name": self._data_type,
                        "data_path": _data_name,
                        "passage_idx": _idx_l,
                        "file_offset": _tell_pos,
                    }
                    _line_strip = _line.strip()
                    if len(_line_strip) > 0:  # skip the empty line
                        yield _line.strip(), _meta_data
                    _tell_pos = fp.tell()
                    _idx_l += 1
                    _line = fp.readline()

    def sentence_iter_simple(self):
        for _passage, _meta_data in self.passage_iter():
            _sent_list = nltk.sent_tokenize(_passage)
            for _idx_sent, _sent in enumerate(_sent_list):
                _sent_meta_data = copy.deepcopy(_meta_data)
                _sent_meta_data["sentence_idx"] = _idx_sent
                yield _sent, _sent_meta_data

    def sentence_iter(self, use_lemma=True, num_parallels=1, buffer_size=1000):
        # introduce multi-processing to this function with passage buffer
        if self._spacy_nlp is None:
            self._load_spacy_nlp()

        assert buffer_size > 1 and num_parallels >= 1
        passage_buffer = []
        max_buffer_size = buffer_size * num_parallels

        def _processor(_passage_list, _spacy_nlp):
            _res_sent_list = []
            for _passage, _meta_data in _passage_list:
                try:
                    _doc = _spacy_nlp(_passage)
                    # sentence split
                    for _idx_sent, _sent in enumerate(_doc.sents):
                        _sent_meta_data = copy.copy(_meta_data)
                        _sent_meta_data["sentence_start"] = _sent.start_char
                        _sent_meta_data["sentence_end"] = _sent.end_char
                        _sent_meta_data["sentence_idx"] = _idx_sent

                        if use_lemma:
                            _lemma_list = []
                            _lemma_start_list = []
                            for _stoken in _sent:
                                _proc_lemma = _stoken.lemma_.strip()
                                if len(_proc_lemma) > 0:
                                    _lemma_list.append(_proc_lemma)
                                    _lemma_start_list.append(_stoken.idx)
                            _sent_meta_data["lemma"] = " ".join(_lemma_list)
                            _sent_meta_data["lemma_start"] = _lemma_start_list

                        _res_sent_list.append((_sent.text, _sent_meta_data))
                except ValueError:
                    print("Got a ValueError and skipped, from passage \"{}\"".format(_passage))
            return _res_sent_list

        def _process_buffer(_passage_buffer):
            if num_parallels == 1:
                _res_sent_list = _processor(passage_buffer, self._spacy_nlp)
            else:
                _res_sent_lists = multiprocessing_map(
                    _processor,
                    dict_args_list=[
                        {"_passage_list": _sub_passage_list, "_spacy_nlp": self._spacy_nlp}
                        for _sub_passage_list in split_to_lists(passage_buffer, num_parallels)],
                    num_parallels=num_parallels
                )
                _res_sent_list = combine_from_lists(_res_sent_lists)

            return _res_sent_list

        for _passage, _meta_data in self.passage_iter():
            # if _meta_data["passage_idx"] % 1000 == 0:
            #     print("Progress {}".format(_meta_data["passage_idx"]))

            passage_buffer.append((_passage, _meta_data))
            if len(passage_buffer) >= max_buffer_size:
                res_sent_list = _process_buffer(passage_buffer)
                for _sent_text, _sent_meta in res_sent_list:
                    yield _sent_text, _sent_meta
                passage_buffer = []
        # if there are also some passages in buffer
        if len(passage_buffer) > 0:
            res_sent_list = _process_buffer(passage_buffer)
            for _sent_text, _sent_meta in res_sent_list:
                yield _sent_text, _sent_meta

class ArcRawTextLoader(RawTextLoader):
    def __init__(self, data_type, data_dir):
        super(ArcRawTextLoader, self).__init__(data_type, data_dir)
        self._data_path_list = [
            (os.path.join(self._data_dir, "ARC_Corpus.txt"), "ARC_Corpus.txt")
        ]

    def passage_iter(self):
        for _p, _meta in super(ArcRawTextLoader, self).passage_iter():
            _search_idx = 0
            while True:
                _res_idx = _p.find("- ", _search_idx)
                if _res_idx < 0:
                    break
                _flag = False
                if _res_idx > 0 and (_res_idx + 2) < len(_p) and is_word(_p[_res_idx - 1]) and is_lower(
                        _p[_res_idx + 2]):
                    if ((_res_idx + 4) >= len(_p) or _p[_res_idx + 2:_res_idx + 5] != "or ") and \
                            ((_res_idx + 5) >= len(_p) or _p[_res_idx + 2:_res_idx + 6] != "and "):
                        if _res_idx < 4 or not (_p[_res_idx - 4] == "-" and is_capital(_p[_res_idx - 3:_res_idx])):
                            _p = _p[:_res_idx] + _p[_res_idx + 2:]
                            _search_idx = _res_idx
                            _flag = True
                if not _flag:
                    _search_idx = _res_idx + 2
            yield _p, _meta

    def sentence_iter_simple(self):
        for _passage, _meta_data in self.passage_iter():
            _meta_data["sentence_idx"] = 0
            yield _passage, _meta_data


class OpenbookqaRawTextLoader(RawTextLoader):
    def __init__(self, data_type, data_dir):
        super(OpenbookqaRawTextLoader, self).__init__(data_type, data_dir)
        self._data_path_list = [
            (os.path.join(self._data_dir, "Additional", "crowdsourced-facts.txt"),
             os.path.join("Additional", "crowdsourced-facts.txt"))
        ]


class WikipediaRawTextLoader(RawTextLoader):
    def __init__(self, data_type, data_dir):
        super(WikipediaRawTextLoader, self).__init__(data_type, data_dir)
        self._data_path_list = [
            (os.path.join(self._data_dir, "wiki_pure_text.txt"), "wiki_pure_text.txt")
        ]


class BookcorpusRawTextLoader(RawTextLoader):
    def __init__(self, data_type, data_dir):
        super(BookcorpusRawTextLoader, self).__init__(data_type, data_dir)
        self._data_path_list = []
        for _path in get_data_path_list(os.path.join(self._data_dir,"raw_data")):
            assert _path.startswith(self._data_dir)
            _name = _path[len(self._data_dir):]
            self._data_path_list.append((_path, _name))


class OmcsRawTextLoader(RawTextLoader):
    def __init__(self, data_type, data_dir):
        super(OmcsRawTextLoader, self).__init__(data_type, data_dir)
        self._data_path_list = [
            (os.path.join(data_dir, "omcs-sentences-more.txt.clean.txt"), "omcs-sentences-more.txt.clean.txt"),
        ]

    def sentence_iter_simple(self):
        for _passage, _meta_data in self.passage_iter():
            _meta_data["sentence_idx"] = 0
            yield _passage, _meta_data

