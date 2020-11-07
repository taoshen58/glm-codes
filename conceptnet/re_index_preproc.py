import json
import argparse
import spacy
import os
from tqdm import tqdm

def main():
    # nlp = spacy.load("en", disable=['parser', 'tagger', 'ner', 'textcat'])

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        raise AssertionError

    index_list = []  # id, offset, n_edges, n_tokens
    with open(args.input_path, encoding="utf-8") as rfp:
        _tell_pos = rfp.tell()
        _idx_l = 0
        _line = rfp.readline()
        while _line:
            if _idx_l % 10000 == 0:
                print(_idx_l)
            # read
            _preproc = json.loads(_line)
            # id
            _id = _preproc[0]
            _offset = _tell_pos
            _n_edges = len(_preproc[3])
            _n_tokens = len(_preproc[1].split(" "))
            index_list.append([_id, _offset, _n_edges, _n_tokens])
            _tell_pos = rfp.tell()
            _idx_l += 1
            _line = rfp.readline()

    with open(args.output_path, "w", encoding="utf-8") as wfp:
        json.dump(index_list, wfp)


def combine_processed_file(path1, path2, out_path):  #
    ofp = open(out_path, "w", encoding="utf-8")
    # get 1st file max id
    last_line = None
    with open(path1, encoding="utf-8") as fp:
        for _line in tqdm(fp):
            last_line = _line
            ofp.write(_line)
    assert last_line is not None and len(last_line.strip()) > 0
    last_data = json.loads(last_line)
    start_idx = int(last_data[0].split("-")[-1]) + 1

    print("Start idx is {}".format(start_idx))

    with open(path2, encoding="utf-8") as rfp:
        for _line in tqdm(rfp):
            _org_data = json.loads(_line)
            _data_name, _org_idx = _org_data[0].split("-")

            _new_data = _org_data
            _new_idx = start_idx + int(_org_idx)
            _new_id = "{}-{}".format(_data_name, _new_idx)
            _new_data[0] = _new_id
            ofp.write(json.dumps(_new_data))
            ofp.write(os.linesep)
    ofp.close()


if __name__ == '__main__':
    main()




