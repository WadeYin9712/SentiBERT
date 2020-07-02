from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve

import random
# from random import random, randrange, randint, shuffle, choice
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json
import collections
import pickle as pkl
import os

np.random.seed(11456)
random.seed(11456)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = random.randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + random.randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, sentiword):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    random.shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        
        word = ""
        for index in index_set:
            if tokens[index].startswith("##"):
                word += tokens[index][2:]
            else:
                word = tokens[index]
                
        if word in sentiword:
            if random.random() < 0.2:
                for index in index_set:
                    covered_indexes.add(index)
        
                    masked_token = None
                    # 80% of the time, replace with [MASK]
                    if random.random() < 0.8:
                        masked_token = "[MASK]"
                    else:
                        # 10% of the time, keep original
                        if random.random() < 0.5:
                            masked_token = tokens[index]
                        # 10% of the time, replace with random word
                        else:
                            masked_token = random.choice(vocab_list)
                    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                    tokens[index] = masked_token
        else:
            if random.random() < 0.15:
                for index in index_set:
                    covered_indexes.add(index)
        
                    masked_token = None
                    # 80% of the time, replace with [MASK]
                    if random.random() < 0.8:
                        masked_token = "[MASK]"
                    else:
                        # 10% of the time, keep original
                        if random.random() < 0.5:
                            masked_token = tokens[index]
                        # 10% of the time, replace with random word
                        else:
                            masked_token = random.choice(vocab_list)
                    masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
                    tokens[index] = masked_token
            

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def convert_to_graph(graph_a, max_len, len_cur, tokens_a_len, total_len):
    len_cur = 0
    len_cur_phrase = total_len
    # max_rc_total = -1
        
    count = 0
    for graph in graph_a:
        max_rc = -1
        
        graph_new_a = np.zeros((max_len * 2, max_len * 2))
        for i in range(max_len * 2):
            graph_new_a[i][i] = 1
        
        for i in range(max_len * 2):
            for j in range(max_len * 2):
                if graph[i][j] == 1 and i != j:
                    # print('edge 1', i,j)
                    new_i = -1
                    new_j = -1
                    if i >= tokens_a_len[count] and j < tokens_a_len[count]:
                        graph_new_a[len_cur_phrase + i - tokens_a_len[count]][j + len_cur] = 1
                        new_i = len_cur_phrase + i - tokens_a_len[count]
                        new_j = j + len_cur
                        # print('edge 2', new_i, new_j)
                    elif i < tokens_a_len[count] and j >= tokens_a_len[count]:
                        graph_new_a[i + len_cur][len_cur_phrase + j - tokens_a_len[count]] = 1
                        new_i = i + len_cur
                        new_j = len_cur_phrase + j - tokens_a_len[count]
                        # print('edge 2', new_i, new_j)
                    elif i >= tokens_a_len[count] and j >= tokens_a_len[count]:
                        graph_new_a[len_cur_phrase + i - tokens_a_len[count]][len_cur_phrase + j - tokens_a_len[count]] = 1
                        new_i = len_cur_phrase + i - tokens_a_len[count]
                        new_j = len_cur_phrase + j - tokens_a_len[count]
                        # print('edge 2', new_i, new_j)

                    if i > max_rc:
                        max_rc = i
                    if j > max_rc:
                        max_rc = j

        len_cur += tokens_a_len[count]
        len_cur_phrase += max_rc + 1 - tokens_a_len[count]
        # print('phrase', max_rc + 1 - tokens_a_len[count])
        # print('hey', len_cur, len_cur_phrase)
        count += 1

    # print('-------------------')
    return graph_new_a, len_cur_phrase

def merge(graph_new_a, graph_new_b, max_word_a, max_word_b, max_phrase_a):
    max_len = 128
    graph_new = np.zeros((max_len * 2, max_len * 2))
    delta = max_word_a + max_word_b

    for i in range(max_len * 2):
        graph_new[i][i] = 1
    
    max_rc = -1
    max_rc_a = -1
    max_rc_b = -1
    max_total = -1
    for i in range(max_len * 2):
        for j in range(max_len * 2):
            if graph_new_a[i][j] == 1 and i < j:
                if i < max_word_a:
                    graph_new[i + 1][j - max_word_a + delta + 3] = 1
                    graph_new[j - max_word_a + delta + 3][i + 1] = 1
                else:
                    graph_new[i - max_word_a + delta + 3][j - max_word_a + delta + 3] = 1
                    graph_new[j - max_word_a + delta + 3][i - max_word_a + delta + 3] = 1
                    
                if j > max_rc_a:
                    max_rc_a = j

            if graph_new_b[i][j] == 1 and i < j:
                if i < max_word_b:
                    # print(i + max_word_a + 2, j - max_word_b + delta + 3 + max_phrase_a - max_word_a)
                    graph_new[i + max_word_a + 2][j - max_word_b + delta + 3 + max_phrase_a - max_word_a] = 1
                    graph_new[j - max_word_b + delta + 3 + max_phrase_a - max_word_a][i + max_word_a + 2] = 1
                else:
                    graph_new[i - max_word_b + delta + 3 + max_phrase_a - max_word_a][j - max_word_b + delta + 3 + max_phrase_a - max_word_a] = 1
                    graph_new[j - max_word_b + delta + 3 + max_phrase_a - max_word_a][i - max_word_b + delta + 3 + max_phrase_a - max_word_a] = 1

                if j > max_rc_b:
                    max_rc_b = j
                    max_rc = j
                    max_total = j - max_word_b + delta + 3 + max_phrase_a - max_word_a

    return graph_new, max_rc - max_word_b + delta + 3 + max_phrase_a - max_word_a + 1

def merge_span(span_a, tokens_a):
    max_len = 128
    span_new = np.zeros((max_len, max_len))
    
    max_a = -1
    for i in range(max_len):
        s = 0
        for j in range(max_len):
            s += span_a[i][j]  
                    
        if s == 0:
            break

    return span_a, True

def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, doc_total, graph_label, span, span_3, preprocessed, sentiword):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
        
    # print(graph_label[15])

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    max_len = 128
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        flag = 0
        
        if i == len(document) - 1:
            for j in range(len(document)):
                current_chunk = document
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = j
    
                    tokens_a = []
                    tokens_a_len = []
                    graph_label_a = []
                    len_a = 0
                    span_a = np.zeros((max_len, max_len))
                    span_a_3 = np.zeros((max_len, max_len * 2))
                            
                    if len(current_chunk[a_end]) <= target_seq_length:
                        tokens_a.extend(current_chunk[a_end])
                        len_a += len(current_chunk[a_end])
                        tokens_a_len.append(len(current_chunk[a_end]))
                        sen_tmp = ""
                        for word in current_chunk[a_end]:
                            sen_tmp += word + " "
                        graph_label_a += graph_label[doc_total[sen_tmp]]
                        span_a = span[doc_total[sen_tmp]]
                        span_a_3 = span_3[doc_total[sen_tmp]]
                    else:
                        tokens_a.extend(current_chunk[a_end][:target_seq_length])
                        len_a += target_seq_length
                        tokens_a_len.append(target_seq_length)
                        sen_tmp = ""
                        for word in current_chunk[a_end]:
                            sen_tmp += word + " "
                        graph_label_a += graph_label[doc_total[sen_tmp]]
                        span_a = span[doc_total[sen_tmp]]
                        span_a_3 = span_3[doc_total[sen_tmp]]
                    
                    len_cur = 0
                    max_rc = max_len
                    is_random_next = True
    
                    if flag == 0:
                        len_cur = 0
                        span_new, judge = merge_span(span_a, tokens_a)
                        
                        if judge == False:
                            continue
                        
                        graph_label_total = []
                        graph_label_total = graph_label_a
                        graph_label_total += [-1] * (max_len - len(graph_label_total))
    
                        assert len(tokens_a) >= 1
    
                        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                        # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                        # They are 1 for the B tokens and the final [SEP]
                        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
    
                        tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                            tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, sentiword)
    
                        instance = {
                            "tokens": tokens,
                            "segment_ids": segment_ids,
                            "is_random_next": is_random_next,
                            "masked_lm_positions": masked_lm_positions,
                            "masked_lm_labels": masked_lm_labels,
                            "phrase_len": max_rc,
                            "graph_label": graph_label_total,
                            "span": span_new,
                            "span_3": span_a_3
                        }
                        instances.append(instance)
    
                    flag = 0
        i += 1

    return instances

def concatenation(s, gold_word_list):
    word_list = []
    word = ""
    corre = []
    normal = []
    for i in range(len(s)):
        normal.append(i)

    # tokens = tokenizer.tokenize(s)
    tokens = s
    token_list = []
    st = 0
    st_word = 0
    word = ""
    while st < len(tokens):
        if "##" in tokens[st]:
            tmp = tokens[st][2:]
        else:
            tmp = tokens[st]
        word += tmp
        corre.append(st_word)
        if word == gold_word_list[st_word]:
            token_list.append(word)
            word_list.append(word)
            word = ""
            st_word += 1

        st += 1
                
    return word_list, corre, normal

def build_graph(graph_line, corre, normal, iter_num):
    max_len = 128
    graph_new = np.zeros((max_len*2, max_len*2))
    
    for i in range(max_len*2):
        graph_new[i][i] = 1
    
    word_num = corre[-1] + 1
    token_num = normal[-1] + 1
    
    # normal : [0,1,2,3,4]
    # corre :  [0,1,1,2,3]
    # word_list: [0,1,2,3] 1->4
    
    max_rc = -1
    
    for i in range(len(corre)):
        for j in range(len(graph_line)):
            if graph_line[corre[i]][j] == 1 and j > corre[i]:
                if j > word_num - 1:
                    graph_new[i][j-word_num+token_num] = 1
                    graph_new[j-word_num+token_num][i] = 1
                    
                    if j-word_num+token_num > max_rc:
                        max_rc = j-word_num+token_num
    
    for i in range(len(graph_line)):
        for j in range(len(graph_line)):
            if graph_line[i][j] == 1 and j > word_num-1 and i > word_num-1:
                graph_new[i-word_num+token_num][j-word_num+token_num] = 1
                graph_new[j-word_num+token_num][i-word_num+token_num] = 1  
                
                if j-word_num+token_num > max_rc:
                    max_rc = j-word_num+token_num
                    
                if i-word_num+token_num > max_rc:
                    max_rc = i-word_num+token_num
                
    return graph_new


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()
    
    preprocessed = True
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    span = np.load(os.path.join(args.data_dir, "sstphrase_train_span.npy"))
    span = span.tolist()
    span_3 = np.load(os.path.join(args.data_dir, "sstphrase_train_span_3.npy"))
    span_3 = span_3.tolist()
    sentiment_label = np.load(os.path.join(args.data_dir, "sstphrase_label_train.npy"))
    sentiwordnet = open("SentiWordNet_3.0.0.txt", 'r')
    
    flag = 0
    sentiword = []
    for line in sentiwordnet:
        if flag == 0:
            flag = 1
            continue
        
        sentidef = line.strip().split('\t')
        if len(sentidef) <= 4:
            break
        pos_score = sentidef[2]
        neg_score = sentidef[3]
        word_info = sentidef[4]

        if pos_score != '0' or neg_score != '0':
            word_info = word_info.split(' ')
            for word in word_info:
                end_pos = word.find('#')
                sentiword.append(word[:end_pos])
    
    sentiwordnet.close()

    doc_total = {}
    doc_num = 0
    
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        with args.train_corpus.open() as f:
            doc = []
            pre_pro = []
            concat = []
            iter_num = 0
            iter_num_1 = 0

                
            for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                flag = 1
                line = line.strip()

                if line == "." or line == "!" or line == "?" or line == "":
                    iter_num += 1
                    if len(doc) == 0:
                        doc = []
                    else:
                        docs.add_document(doc)
                        
                    for d in doc:
                        sen_tmp = ""
                        for word in d:
                            sen_tmp += word + ' '
                        doc_total[sen_tmp] = doc_num
                        doc_num += 1
                    
                    doc = []
                    concat.append("")
                    continue
                    
                if flag == 0:
                    continue
                else:
                    #store as one sample
                    if preprocessed == False:
                        if len(line.split('. ')) == 0:
                            if len(sentiment_label[iter_num]) + 1 == len(line.split(" ")):
                                s = tokenizer.tokenize(line)
                                
                                doc.append(s)
                                pre_pro.append(s)
                                word_list, corre, normal = concatenation(s, line.split(" "))
                                iter_num_1 += 1
                                concat.append(word_list)
                                    
                                for d in doc:
                                    sen_tmp = ""
                                    for word in d:
                                        sen_tmp += word + ' '
                                    doc_total[sen_tmp] = doc_num
                                    doc_num += 1
                                        
                                docs.add_document(doc)
                                doc = []
                                concat.append("")
                                
                                continue

                        for s_str in line.split('. '):
                            if len(line.split('. ')) == 1:
                                if len(sentiment_label[iter_num]) + 1 == len(line.split(" ")):
                                    s = tokenizer.tokenize(s_str)
                                    
                                    doc.append(s)
                                    pre_pro.append(s)
                                    word_list, corre, normal = concatenation(s, line.split(" "))
                                    iter_num_1 += 1
                                    concat.append(word_list)
                                    
                                    for d in doc:
                                        sen_tmp = ""
                                        for word in d:
                                            sen_tmp += word + ' '
                                        doc_total[sen_tmp] = doc_num
                                        doc_num += 1
                                        
                                    docs.add_document(doc)
                                    doc = []
                                    concat.append("")
                        
                    else:
                        doc = []
                        tokens = tokenizer.tokenize(line)
                        doc.append(tokens)
                        
                        for d in doc:
                            sen_tmp = ""
                            for word in d:
                                sen_tmp += word + ' '
                            doc_total[sen_tmp] = doc_num
                            doc_num += 1
                                        
                        docs.add_document(doc)
                        doc = []

                iter_num += 1
                
            if doc:
                docs.add_document(doc)  # If the last doc didn't end on a newline, make sure it still gets added
                for d in doc:
                    sen_tmp = ""
                    for word in d:
                        sen_tmp += word + ' '
                    doc_total[sen_tmp] = doc_num
                    doc_num += 1
                    
                doc = []
                
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        # if preprocessed == False:
        #     np.save("graph_token.npy", graph)

        print('docs', len(docs))
        args.output_dir.mkdir(exist_ok=True)
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = args.output_dir / f"epoch_{epoch}.json"
            num_instances = 0
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances = create_instances_from_document(
                        docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, doc_total=doc_total, graph_label=sentiment_label, span=span, span_3=span_3, preprocessed=preprocessed, sentiword=sentiword)
                    
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = args.output_dir / f"epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                metrics_file.write(json.dumps(metrics))


if __name__ == '__main__':
    main()
