
# GLM Codes

**Release note:**
* The code is based on Huggingface's [Transformers](https://github.com/huggingface/transformers) with various pre-trained models. Thank [@huggingface](https://github.com/huggingface)
* If you encounter any problem or bug, no hesitate to contact me by email to [tao.shen@student.uts.edu.au](tao.shen@student.uts.edu.au) or opening an issue. 
* This is an init version which would include some bugs caused by re-organizing. 

## Python Env Build with Anaconda

* Please install CUDA10 and add to env variables
* then install python packages with Anaconda virtual env
```
conda create -n venv python=3.6
source activate venv
onda install -y pytorch=1.1 torchvision -c pytorch
conda install -y -c conda-forge spacy=2.1.8; python -m spacy download en
pip install regex networkx tqdm scipy fuzzywuzzy tensorboardX boto3 nltk elasticsearch python-Levenshtein sacremoses scikit-learn
git clone https://github.com/clips/pattern; cd pattern/; python setup.py install; cd ../
```
* Optional but recommend, install `apex` for half/mixed float computation:
```
git clone https://github.com/NVIDIA/apex; cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
```


# Language Model Training using our approach



## Data processing

### Data Preparation

First of all, change the dirs' name in `configs.py`.

#### Download ConceptNet

Download conceptNet from [here](https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz),
and put `assertions.csv` to a separate dir.

#### Download corpora, clean and rename 
* OMCS [here](https://s3.amazonaws.com/conceptnet/downloads/2018/omcs-sentences-more.txt)
    1. please clean all non-english triplet
    2. rename the name of cleaned corpus as `omcs-sentences-more.txt.clean.txt`  
* ARC [here](http://data.allenai.org/arc/)
    1. view downloaded data dir, ensure the corpus named as `ARC_Corpus.txt` 
* Wikipedia
    1. please download and clean Wikipedia follow BERT
    2. rename the name of cleaned corpus as `wiki_pure_text.txt`
* any other raw text corpus

We will upload the propocessed data when the paper is publicly released.


### Processing ConceptNet
Note please rename the conceptnet csv file to `assertions.csv`.

```
python3 conceptnet/pipline.py
```

### Process the data

#### index sentences/passages from corpus for future use
```
python3 data_proc/index_sent.py --data_type omcs --num_workers 16
python3 data_proc/index_sent.py --data_type arc --num_workers 16
python3 data_proc/index_sent.py --data_type wikipedia --num_workers 16
```
Here, wikipedia is only used for provide statistics for stop concept on ConceptNet 

#### index concept for efficiency in downsteams
```
python3 data_proc/index_cn.py --data_type_list omcs,arc,wikipedia --num_workers 20
```

### reindex sentences/passages based on the indexed conceptnet
```
python3 data_proc/index_sent_cn.py --data_type omcs --num_workers 16
python3 data_proc/index_sent_cn.py --data_type arc --num_workers 16
```

We will upload the propocessed data when the paper is publicly released.


## Language Model Training


```
MODEL_OUTDIR="path to a dir"
PRETRAINED_MODEL_CLASS="bert"
PRETRAINED_MODEL_NAME="bert-base-uncased"

python3 main_span_line.py --do_train --learning_rate 3e-5 --num_train_epochs 5 --eval_steps 1000 --fp16 --num_workers 4 --seed 51 \
  --sent_format single --gradient_accumulation_steps 1 --train_batch_size 112 --eval_batch_size 64 \
  --model_class PRETRAINED_MODEL_CLASS --model_name_or_path PRETRAINED_MODEL_NAME --do_lower_case \
  --data_format cn --model_type glm \
  --output_dir $MODEL_OUTDIR
```

Note: during organizing this code, we transmitted `pytorch_transformers` 
to `transformers`, which would lead to some bugs. We will test later.


# Finetuning on downstream Tasks

## CommonsenseQA

```
CQA_PATH="..../rand"
OPT_PARAM="--do_lower_case --adam_epsilon 1e-6 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1"
python3 run_cqa.py --data_dir $CQA_PATH --do_train --do_eval --max_seq_length 80 \
    $OPT_PARAM \
    --learning_rate 5e-5 --max_steps 3400 \
    --eval_steps 500 --train_batch_size 12 \
    --eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --model_class PRETRAINED_MODEL_CLASS --model_name_or_path $MODEL_OUTDIR \
    --output_dir ${MODEL_OUTDIR}-cqa \
    --seed 42 \
    --fp16
```

The evaluation and prediction results will be shown in `${MODEL_OUTDIR}-cqa`


## WN18RR

### Got the data From [KG-BERT](https://arxiv.org/abs/1909.03193)

```
cd ANY_DIR
git clone https://github.com/yao8839836/kg-bert
```
The data is located at `kg-bert/data`

```
python3 link_prediction.py --model_class bert --do_train --do_eval \
    --warmup_proportion 0.1 --learning_rate 5e-5 --num_train_epochs 5. --dataset WN18RR \
    --max_seq_length 64 --gradient_accumulation_steps 1 --train_batch_size 32 \
    --eval_batch_size 128 --logging_steps 100 --eval_steps -1 --save_steps 2000 \
    --model_name_or_path $MODEL_OUTDIR --do_lower_case --output_dir ${MODEL_OUTDIR}-WN18RR \
    --num_worker 12 --seed 42 --fp16 \
    --data_dir /path-to-kg-bert-data-dir
```
Note dev evaluation during traning and only save the last checkpoint for testing
```
python3 link_prediction.py \
    --model_class bert --do_lower_case --do_prediction \
    --dataset WN18RR --max_seq_length 64 --eval_batch_size 256 \
    --model_name_or_path ${MODEL_OUTDIR}-WN18RR --output_dir ${MODEL_OUTDIR}-WN18RR-eval \
    --num_worker 4 --fp16 \
    --data_dir /path-to-kg-bert-data-dir \
    --prediction_part 0,1
```
The test result will be both printed and saved to `${MODEL_OUTDIR}-WN18RR-eval`

## WN11

Data fetch procedure is same as WN18RR

```
OPT_PARAM="--do_lower_case --adam_epsilon 1e-6 --max_grad_norm 1. --weight_decay 0.01 --warmup_proportion 0.1"
DATA_PARAM="--max_seq_length 32 --gradient_accumulation_steps 1"
python3 triplet_cls.py --do_train --do_eval \
    $DATA_PARAM --dataset WN11 \
    $OPT_PARAM \
    --learning_rate 5E-5 --num_train_epochs 5 \
    --eval_steps 1000 --train_batch_size 32 \
    --eval_batch_size 64 \
    --model_class $PRETRAINED_MODEL_CLASS --model_name_or_path $MODEL_OUTDIR \
    --output_dir ${MODEL_OUTDIR}-WN11 \
    --seed $SEED --num_worker 8
```

The test result will saved to `${MODEL_OUTDIR}-WN11`

## Cite this paper

```
@article{shen2020exploiting,
  title={Exploiting Structured Knowledge in Text via Graph-Guided Representation Learning},
  author={Shen, Tao and Mao, Yi and He, Pengcheng and Long, Guodong and Trischler, Adam and Chen, Weizhu},
  journal={arXiv preprint arXiv:2004.14224},
  year={2020}
}
```



