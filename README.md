# SentiBERT
Code for *SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics* (ACL'2020).
https://arxiv.org/abs/2005.04114

## Model Architecture
<p align="center">
    <img src="model.png" height="300" />
</p> 

## Requirements
### Environment
```
* Python == 3.6.10
* Pytorch == 1.1.0
* CUDA == 10.2
* NVIDIA GeForce GTX 1080 Ti
* HuggingFaces Pytorch (also known as pytorch-pretrained-bert & transformers)
* Stanford CoreNLP (stanford-corenlp-full-2018-10-05)
* Numpy, Pickle, Tqdm, Scipy, etc. (See requirement.txt)
```

### Datasets
Datasets include:
```
* SST-phrase
* SST-5 (almost the same with SST-phrase)
* SST-3 (almost the same with SST-phrase)
* SST-2
* Twitter Sentiment Analysis (SemEval 2017 Task 4)
* EmoContext (SemEval 2019 Task 3)
* EmoInt (Joy, Fear, Sad, Anger) (SemEval 2018 Task 1c)
```

### File Architecture (Selected important files)
```
-- /examples/run_classifier_new.py                                  ---> start to run
-- /examples/run_classifier_dataset_utils_new.py                    ---> input preprocessed files to SentiBERT
-- /pytorch-pretrained-bert/modeling_new.py                         ---> detailed model architecture
-- /examples/lm_finetuning/pregenerate_training_data_sstphrase.py   ---> generate pretrained epoches
-- /examples/lm_finetuning/finetune_on_pregenerated_sstphrase.py    ---> pretrain on generated epoches
-- /stanford-corenlp-full-2018-10-05/xxx_st.py (under construction) ---> preprocess raw text and constituency tree
-- /transformers (under construction)                               ---> RoBERTa part
```

## Get Started
### Preparing Environment
```
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:XX/SentiBERT/
export PYTHONPATH=$PYTHONPATH:XX/
```
### Preprocessing (under construction)
1. Split the raw text and golden labels of sentiment/emotion datasets by yourselves into `xxx_train\dev\test_text.txt` and `xxx_label_train\dev\test.npy`, assuming that `xxx` represents task name.
2. Put the files into `/stanford-corenlp-full-2018-10-05/`. To get binary constituency trees, run
```
java -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,parse,sentiment -file xxx_train\dev\test_text.txt -outputFormat json -ssplit.eolonly true -tokenize.whitespace true
```
3. Run `/stanford-corenlp-full-2018-10-05/xxx_st.py` to transform the tree structure into matrices `/glue_data/xxx/xxx_train\dev\test_span.npy` and `/glue_data/xxx/xxx_train\dev\test_span_3.npy`. The first matrix is used as mask in the first layer of our attention mechanism. The second matrix is used as mask in the second layer.

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
The pre-trained parameters were released here. [[Google Drive]](https://drive.google.com/file/d/1VPKeB_FjrAiSYfEi-F72wtZkaYQD-l7Q/view?usp=sharing)

## Fine-tuning 
Run run_classifier_new.py directly as follows:
```
CUDA_VISIBLE_DEVICES=7 python run_classifier_new.py \
  --task_name xxx \                              ---> task name
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir /glue_data/xxx \                    ---> the same name as task_name
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size xxx \
  --learning_rate xxx \
  --num_train_epochs xxx \                                                          
  --domain xxx \                                 ---> "joy", "sad", "fear" or "anger". Used in EmoInt task
  --output_dir /results/xxx \                    ---> the same name as task_name
  --seed xxx \
  --para xxx                                     ---> "SentiBERT" or "BERT": pretrained SentiBERT or BERT
```

## Checkpoints
For reproducity and usability, we provide checkpoints and the original training settings to help you reproduce:
 * SST-phrase [[Google Drive]](https://drive.google.com/file/d/17if73T2bbOhAqG41RxkJ7HxfusUjm-hg/view?usp=sharing)
 * SST-5 [[Google Drive]](https://drive.google.com/file/d/17if73T2bbOhAqG41RxkJ7HxfusUjm-hg/view?usp=sharing)
 * SST-2 [[Google Drive]](https://drive.google.com/file/d/1JiPv5Wv56A6JccgLBS_-1LSFna63iL5T/view?usp=sharing)
 * SST-3 [[Google Drive]](https://drive.google.com/file/d/1XsmcGyotHfVABaewxY_EcESR7Dku66Ln/view?usp=sharing)
 * EmoContext [[Google Drive]](https://drive.google.com/file/d/1rpO5rmBY6rX6rbZuyCoGV40J0JpcPo2x/view?usp=sharing)
 * EmoInt:
     * Joy [[Google Drive]](https://drive.google.com/file/d/1OTGBRlcWISzH2bl2aKs9YMN2XQKJ9bJj/view?usp=sharing)
     * Fear [[Google Drive]](https://drive.google.com/file/d/1b8db93qOOpSMJjSRI1pJUumdRS4X83Xg/view?usp=sharing)
     * Sad [[Google Drive]](https://drive.google.com/file/d/1En9Vcn1JdG8NyxfuQbUCSaToSpUu8Oxp/view?usp=sharing)
     * Anger [[Google Drive]](https://drive.google.com/file/d/1vm0cSyqTbm41qe_bVtVGkt0mSYTsdQko/view?usp=sharing)
 * Twitter Sentiment Analysis [[Google Drive]](https://drive.google.com/file/d/16CQh9WdqhzeWfkxToZholnOqmaNEnayG/view?usp=sharing)

The implementation details and results are shown below:
**Note: BERT* denotes BERT w/ Mean pooling.**
<table>
  <tr>
    <th>Models</th>
    <th>Batch Size</th>
    <th class="tg-0pky">Learning Rate</th>
    <th class="tg-0pky">Epochs</th>
    <th class="tg-0pky">Seed</th>
    <th class="tg-0pky">Results</th> 
  </tr>
  <tr>
    <td colspan="6">SST-phrase</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">**68.89**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">64.76</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">SST-5</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">**56.64**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">49.13</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">SST-2</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">**93.02**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">30</td>
    <td class="tg-0pky">92.08</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">SST-3</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">77</td>
    <td class="tg-0pky">**77.84**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">77</td>
    <td class="tg-0pky">72.71</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">EmoContext</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">**75.85**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">73.64</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">EmoInt</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">16</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">4 or 5</td>
    <td class="tg-0pky">77</td>
    <td class="tg-0pky">**67.24**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT</td>
    <td class="tg-0pky">16</td>
    <td class="tg-0pky">2e-5</td>
    <td class="tg-0pky">4 or 5</td>
    <td class="tg-0pky">77</td>
    <td class="tg-0pky">64.79</td>
  </tr>
  <tr>
    <td class="tg-baqh" colspan="6">Twitter Sentiment Analysis</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">6e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">45</td>
    <td class="tg-0pky">**70.1**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT</td>
    <td class="tg-0pky">32</td>
    <td class="tg-0pky">6e-5</td>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">45</td>
    <td class="tg-0pky">69.7</td>
  </tr>
</table>

## Analysis
Here we provide analysis implementation in our paper. We will focus on the evaluation of 
* local difficulty
* global difficulty
* negation
* contrastive relation

In preprocessing part, we provide implementation to extract related information in the test set of SST-phrase and store them in 
```
-- /glue_data/sstphrase/swap_test_new.npy                   ---> global difficulty
-- /glue_data/sstphrase/edge_swap_test_new.npy              ---> local difficulty
-- /glue_data/sstphrase/neg_new.npy                         ---> negation
-- /glue_data/sstphrase/but_new.npy                         ---> contrastive relation
```
In `simple_accuracy_phrase()`, we will provide statistical details and evaluate for each metric.

Some of the analysis results based on our provided checkpoints are selected and shown below:
<table>
  <tr>
    <th class="tg-0pky">Models</th>
    <th class="tg-0pky">Results</th> 
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">Local Difficulty</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">**[85.41, 60.69, 49.03]**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">[82.42, 55.64, 32.19]</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">Negation</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">**[78.34, 76.30, 72.77]**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">[74.55, 71.36, 69.72]</td>
  </tr>
  <tr>
    <td class="tg-c3ow" colspan="2">Contrastive Relation</td>
  </tr>
  <tr>
    <td class="tg-0pky">SentiBERT</td>
    <td class="tg-0pky">**39.24**</td>
  </tr>
  <tr>
    <td class="tg-0pky">BERT*</td>
    <td class="tg-0pky">27.85</td>
  </tr>
</table>

## Acknowledgement
Here we would like to thank for BERT/RoBERTa implementation of HuggingFace and sentiment tree parser of Stanford CoreNLP. Also, thanks for the dataset release of SemEval. To confirm the privacy rule of SemEval task organizer, we only choose the publicable datasets of each task.

## Citation
Please cite our ACL paper if this repository inspired your work.
```
@inproceedings{yin2020sentibert,
  author    = {Yin, Da and Meng, Tao and Chang, Kai-Wei},
  title     = {{SentiBERT}: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics},
  booktitle = {Proceedings of the 58th Conference of the Association for Computational Linguistics, {ACL} 2020, Seattle, USA},
  year      = {2020},
}
```
    
## Contact
* Due to the difference of environment, the results will be a bit different. If you have any questions regarding the code, please create an issue or contact the [owner](https://github.com/WadeYin9712) of this repository.
