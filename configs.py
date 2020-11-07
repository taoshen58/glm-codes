import os

USER_HOME = os.getenv("HOME")

# index dir
index_sent_cache_dir = USER_HOME + "/a_path_to"

# concpet net
conceptnet_dir = USER_HOME + "/a_path_to"
rel2idx_path = os.path.join(conceptnet_dir, "assertions.csv_en.csv_rel2idx.txt")
triplet2template_path = os.path.join(conceptnet_dir, "template.json")

# raw corpus

omcs_dir = USER_HOME + "/a_path_to"
arc_dir = USER_HOME + "/a_path_to"
wikipedia_dir = USER_HOME + "/a_path_to"
openbookqa_dir = USER_HOME + "/a_path_to"
bookcorpus_dir = USER_HOME + "/a_path_to"

# KG-BERT's DATAIDR

kgbert_data_dir = USER_HOME + "/a_path_to"















