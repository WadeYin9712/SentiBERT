# SentiBERT
Code for *SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics* (ACL'2020).
https://arxiv.org/abs/2005.04114

## Model Architecture
<p align="center">
    <img src="model.png" height="300" />
</p> 

## Requirements
### Environment
* Python == 3.6.10
* Pytorch == 1.1.0
* CUDA == 10.2
* NVIDIA GeForce GTX 1080 Ti
* HuggingFaces Pytorch (also known as pytorch-pretrained-bert & transformers)
* Stanford CoreNLP (stanford-corenlp-full-2018-10-05)
* Numpy, Pickle, Tqdm, Scipy, etc. (See requirement.txt)

### Datasets
Datasets include:
* SST-phrase
* SST-5 (almost the same with SST-phrase)
* SST-3 (almost the same with SST-phrase)
* SST-2
* Twitter Sentiment Analysis (SemEval 2017 Task 4)
* EmoContext (SemEval 2019 Task 3)
* EmoInt (Joy, Fear, Sad, Anger) (SemEval 2018 Task 1c)

### File Architecture (Selected important files)
```
-- /examples/run_classifier_new.py                                  ---> start to run
-- /examples/run_classifier_dataset_utils_new.py                    ---> input preprocessed files to SentiBERT
-- /pytorch-pretrained-bert/modeling_new.py                         ---> detailed model architecture
-- /examples/lm_finetuning/pregenerate_training_data_sstphrase.py   ---> generate pretrained epoches
-- /examples/lm_finetuning/finetune_on_pregenerated_sstphrase.py    ---> pretrain on generated epoches
-- /stanford-corenlp-full-2018-10-05/xxx_st.py (under construction) ---> preprocess raw text and constituency tree
```

## Get Started
### Preprocessing (under construction)
1. Split the raw text and golden labels of sentiment/emotion datasets by yourselves into `xxx_train\dev\test_text.txt` and `xxx_label_train\dev\test.npy`, assuming that `xxx` represents task name.
2. Put the files into `/stanford-corenlp-full-2018-10-05/`. To get binary constituency trees, run
```
java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,parse,sentiment -file xxx_train\dev\test_text.txt -outputFormat json -ssplit.eolonly true -tokenize.whitespace true
```
3. Run `/stanford-corenlp-full-2018-10-05/xxx_st.py` to transform the tree structure into matrices `/glue_data/xxx/xxx_train\dev\test_span.npy` and `/glue_data/xxx/xxx_train\dev\test_span_3.npy`. The first matrix is used as mask in the first layer of our attention mechanism. The second matrix is used as mask in the second layer. Will first publish them via Google Drive.

## Pretraining
1. Generate epoches for preparation
```
python3 pregenerate_training_data_sstphrase.py \
        --train_corpus /glue_data/sstphrase/sstphrase_train_text.txt \
        --bert_model bert-base-uncased \
        --do_lower_case \
        --output_dir /training_sstphrase \
        --epochs_to_generate 3 \
        --max_seq_len 128 \
```
2. Pretrain the generated epoches
```
CUDA_VISIBLE_DEVICES=7 python3 finetune_on_pregenerated_sstphrase.py \
        --pregenerated_data /training_sstphrase \
        --bert_model bert-base-uncased \
        --do_lower_case \
        --output_dir /results/sstphrase_pretrain \
        --epochs 3
```
