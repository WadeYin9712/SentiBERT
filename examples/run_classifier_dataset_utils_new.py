# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
import pickle as pkl

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, span=None, span_3=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a.strip()
        self.text_b = text_b
        self.label = label
        self.span = span
        self.span_3 = span_3


class InputFeatures_phrase(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, phrase_mask, segment_ids, label_id, span, span_3):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.phrase_mask = phrase_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.span = span
        self.span_3 = span_3
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, span, span_3):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.span = span
        self.span_3 = span_3


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "sst2_train_text_new.txt"))
        label = np.load(os.path.join(data_dir, "sst2_label_train.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "sst2_train_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "sst2_train_span_3.npy"))
        else:
            span = None
            span_3 = None

        return self._create_examples(lines, label, span, span_3, "train")

    def get_dev_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "sst2_dev_text_new.txt"))
        label = np.load(os.path.join(data_dir, "sst2_label_dev.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "sst2_dev_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "sst2_dev_span_3.npy"))
        else:
            span = None
            span_3 = None

        return self._create_examples(lines, label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        label_sst = label
        span_sst = span
        span_3_sst = span_3

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line
            label = label_sst[i]

            if span is not None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label, span=span_sst[i], span_3=span_3_sst[i]))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, span=None, span_3=None, label=label))
        return examples

class SstPhraseProcessor(DataProcessor):
    """Processor for the SST-phrase data set."""

    def get_train_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "train_text.txt"))
        graph_label = np.load(os.path.join(data_dir, "sentiment_train.npy"))
        span = np.load(os.path.join(data_dir, "span_train_new.npy"))
        span_3 = np.load(os.path.join(data_dir, "span_train_new_3.npy"))

        return self._create_examples(lines, graph_label, span, span_3, "train")

    def get_dev_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "test_text.txt"))
        graph_label = np.load(os.path.join(data_dir, "sentiment_test.npy"))
        span = np.load(os.path.join(data_dir, "span_test_new.npy"))
        span_3 = np.load(os.path.join(data_dir, "span_test_new_3.npy"))

        return self._create_examples(lines, graph_label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, graph_label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        label_sst = graph_label
        span_sst = span
        span_sst_3 = span_3

        examples = []
        i = 0
        for line in lines:
            guid = "%s-%s" % (set_type, i)
            text_a = line
            label = []
            for j in range(len(label_sst[i])):
                if int(label_sst[i][j]) >= 0 and int(label_sst[i][j]) <= 4:
                    label.append(int(label_sst[i][j]))
                else:
                    label.append(-1)
                    
            span_tmp = span_sst[i]
            span_3_tmp = span_sst_3[i]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, 
                             span=span_tmp,
                             span_3=span_3_tmp,
                             label=label))

            i += 1    
            
        return examples
    
class Sst3Processor(DataProcessor):
    """Processor for the SST-3 data set."""

    def get_train_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "train_text.txt"))
        graph_label = np.load(os.path.join(data_dir, "sentiment_train.npy"))
        span = np.load(os.path.join(data_dir, "span_train_new.npy"))
        span_3 = np.load(os.path.join(data_dir, "span_train_new_3.npy"))

        return self._create_examples(lines, graph_label, span, span_3, "train")

    def get_dev_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "test_text.txt"))
        graph_label = np.load(os.path.join(data_dir, "sentiment_test.npy"))
        span = np.load(os.path.join(data_dir, "span_test_new.npy"))
        span_3 = np.load(os.path.join(data_dir, "span_test_new_3.npy"))

        return self._create_examples(lines, graph_label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, graph_label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        label_sst = graph_label
        span_sst = span
        span_sst_3 = span_3

        examples = []
        i = 0
        for line in lines:
            guid = "%s-%s" % (set_type, i)
            text_a = line
            label = []
            for j in range(len(label_sst[i])):
                if int(label_sst[i][j]) == 0 or int(label_sst[i][j]) == 1:
                    label.append(0)
                elif int(label_sst[i][j]) == 3 or int(label_sst[i][j]) == 4:
                    label.append(1)
                elif int(label_sst[i][j]) == 2:
                    label.append(2)
                else:
                    label.append(-1)
                    
            span_tmp = span_sst[i]
            span_3_tmp = span_sst_3[i]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, 
                             span=span_tmp,
                             span_3=span_3_tmp,
                             label=label))

            i += 1    
            
        return examples
    
class TwitterProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def get_train_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "twitter_train_text_new.txt"))
        label = np.load(os.path.join(data_dir, "twitter_label_train.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "twitter_train_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "twitter_train_span_3.npy"))
        else:
            span = None
            span_3 = None

        return self._create_examples(lines, label, span, span_3, "train")

    def get_dev_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "twitter_test_text_new.txt"))
        label = np.load(os.path.join(data_dir, "twitter_label_test.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "twitter_test_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "twitter_test_span_3.npy"))
        else:
            span = None
            span_3 = None    
            
        return self._create_examples(lines, label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0
        for line in lines:
            guid = "%s-%s" % (set_type, i)
            text_a = line.strip()
            text_b = ""
            
            if label[i] == "positive":
                label[i] = "0"
            else:
                if label[i] == "negative":
                    label[i] = "1"
                if label[i] == "neutral":
                    label[i] = "2"
                    
            if span is not None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, span=span[i], span_3=span_3[i], label=str(label[i])))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, span=None, span_3=None, label=str(label[i])))
            
            i += 1

        return examples
    
class EmoContextProcessor(DataProcessor):
    """Processor for the EmoContext data set."""
    def get_train_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "emocontext_train_text_new.txt"))
        label = np.load(os.path.join(data_dir, "emocontext_label_train.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "emocontext_train_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "emocontext_train_span_3.npy"))
        else:
            span = None
            span_3 = None

        return self._create_examples(lines, label, span, span_3, "train")

    def get_dev_examples(self, data_dir, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "emocontext_test_text_new.txt"))
        label = np.load(os.path.join(data_dir, "emocontext_label_test.npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "emocontext_test_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "emocontext_test_span_3.npy"))
        else:
            span = None
            span_3 = None
        
        return self._create_examples(lines, label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        
        i = 0
        
        for line in lines:
            guid = "%s-%s" % (set_type, i)
            text_a = line.strip()
            label_tmp = label[i]
            
            if label_tmp == "angry":
                label_tmp = "0"
            else:
                if label_tmp == "sad":
                    label_tmp = "1"
                if label_tmp == "happy":
                    label_tmp = "2"
            
            if span is not None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b="", span=span[i], span_3=span_3[i], label=label_tmp))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b="", span=None, span_3=None, label=label_tmp))
                
            i += 1

        lines.close()

        return examples
    
class EmoIntProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""
    def get_train_examples(self, data_dir, domain, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "emoint_train_"+domain+"_text_new.txt"))
        label = np.load(os.path.join(data_dir, "emoint_label_train_"+domain+".npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "emoint_train_"+domain+"_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "emoint_train_"+domain+"_span_3.npy"))
        else:
            span = None
            span_3 = None

        return self._create_examples(lines, label, span, span_3, "train")

    def get_dev_examples(self, data_dir, domain, para):
        """See base class."""
        lines = open(os.path.join(data_dir, "emoint_test_"+domain+"_text_new.txt"))
        label = np.load(os.path.join(data_dir, "emoint_label_test_"+domain+".npy"))
        if para == "sentibert":
            span = np.load(os.path.join(data_dir, "emoint_test_"+domain+"_span.npy"))
            span_3 = np.load(os.path.join(data_dir, "emoint_test_"+domain+"_span_3.npy"))
        else:
            span = None
            span_3 = None
        
        return self._create_examples(lines, label, span, span_3, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _create_examples(self, lines, label, span, span_3, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = 0
        
        for line in lines:
            guid = "%s-%s" % (set_type, i)
            text_a = line.strip()
                    
            if span is not None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b="", span=span[i], span_3=span_3[i], label=label[i]))
            else:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b="", span=None, span_3=None, label=label[i]))
                
            i += 1
        
        lines.close()

        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def merge_span(span_a, tokens_a, f):
    max_len = 128
    if f == 0:
        span_new = np.zeros((max_len, max_len))
    else:
        span_new = np.zeros((max_len, max_len*2))
    
    max_a = -1
    for i in range(max_len):
        s = 0
        for j in range(max_len-1):
            s += span_a[i][j]  
        if s == 0:
            if f == 0:
                max_a = i
                break

    if f != 0:
        for i in range(max_len):
            for j in range(max_len * 2):
                if i < len(tokens_a):
                    if j >= max_len * 2 - 1:
                        break
                    span_new[i][j+1] = span_a[i][j]
    else:
        for i in range(max_len):
            for j in range(max_len):
                if i < max_a:
                    if j >= max_len - 1:
                        break
                    span_new[i][j+1] = span_a[i][j]
    
    max_new = -1
    if f == 0:
        for i in range(max_len):
            s = 0
            for j in range(max_len):
                s += span_new[i][j]  
            if s == 0:
                max_new = i
                break

    return span_new, True, max_new

def convert_examples_to_features_phrase(examples, max_seq_length,
                                 tokenizer, output_mode, set_type, data_dir):
    """Loads a data file into a list of `InputBatch`s."""

    if set_type == "dev":
        swap = np.load(os.path.join(data_dir, "swap_test_new.npy"))
        edge = np.load(os.path.join(data_dir, "edge_test_new.npy"))
        edge_swap = np.load(os.path.join(data_dir, "edge_swap_test_new.npy"))
        but = np.load(os.path.join(data_dir, "but_new.npy"))
        neg = np.load(os.path.join(data_dir, "neg_new.npy"))
        text_tmp = open("test_text_new.txt", 'r')
    
        text = []
        for line in text_tmp:
            text.append(line.strip())
    
    features = []
    swap_total = []
    edge_total = []
    edge_swap_total = []
    but_total = []
    neg_total = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 200 == 0:
            print(ex_index)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
            
        label = example.label
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        span_3, judge, max_new = merge_span(example.span_3, tokens_a, 3)
        span, judge, max_new = merge_span(example.span, tokens_a, 0)
        
        if judge == False:
            continue
        
        if set_type == "dev":
            swap_total.append(swap[ex_index])
            edge_total.append(edge[ex_index])
            edge_swap_total.append(edge_swap[ex_index])
            but_total.append(but[ex_index])
            neg_total.append(neg[ex_index])

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        label_id = label
        phrase_mask = []
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_phrase = [0] * (max_seq_length - len(phrase_mask))
        padding_label = [-1] * (max_seq_length - len(label_id))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        phrase_mask += padding_phrase
        label_id += padding_label

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures_phrase(input_ids=input_ids,
                              input_mask=input_mask,
                              phrase_mask=phrase_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              span=span,
                              span_3=span_3))
    
    if set_type == "dev":
        print(len(swap_total), len(edge_total), len(edge_swap_total), len(but_total), len(neg_total))
        np.save("swap_test.npy", swap_total)
        np.save("edge_test.npy", edge_total) 
        np.save("edge_swap_test.npy", edge_swap_total)
        np.save("but_new.npy", but_total) 
        np.save("neg_new.npy", neg_total)
        text_tmp.close()
    
    return features

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, para):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        if para == "sentibert":
            span_a = example.span
            span_a_3 = example.span_3
        else:
            span_a = None
            span_a_3 = None

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        if span_a is not None:
            span_3, judge, max_new = merge_span(span_a_3, tokens_a, 3)
            span, judge, max_new = merge_span(span_a, tokens_a, 0)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
    
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
    
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
    
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
    
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id,
                                  span=span,
                                  span_3=span_3))
        else:
            input_mask = [1] * len(input_ids)
    
            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id,
                                  span=None,
                                  span_3=None)) 
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def global_swap(sum_stat, cor_stat):
    cor_split = [0] * 5
    sum_split = [0] * 5
    result_2 = []
    
    for i in range(5):
        for j in range(5):
            if sum_stat[i*5+j] != 0:
                cor_split[i] += cor_stat[i*5+j]
                sum_split[i] += sum_stat[i*5+j]
                
                
    for i in range(5):
        if sum_split[i] != 0:
            result_2.append(cor_split[i] * 1. / sum_split[i])
            
    return result_2

def local_swap(preds, labels, edge_total, edge_swap_total):
    i = 0
    edge_swap_stat = [0] * 3
    edge_swap_sum = [0] * 3
    for pred in preds:
        label = labels[i]
        edge_tot = edge_total[i]
        edge_swap_tot = edge_swap_total[i]
        assert len(edge_tot) == len(edge_swap_tot)
        t2 = 0
        for edge in edge_tot:
            if label[edge[0]] == pred[edge[0]]:
                edge_swap_stat[edge_swap_tot[t2]] += 1
            edge_swap_sum[edge_swap_tot[t2]] += 1
            
            t2 += 1
        i += 1
        
    return [edge_swap_stat[0] * 1. / edge_swap_sum[0], edge_swap_stat[1] * 1. / edge_swap_sum[1], edge_swap_stat[2] * 1. / edge_swap_sum[2]]

def negation(cor_stat_neg, sum_stat_neg):
    result_4 = []
            
    for i in range(3):
        if i < 2:
            result_4.append(cor_stat_neg[i] * 1. / sum_stat_neg[i])
        else:
            result_4.append((cor_stat_neg[i] + cor_stat_neg[i+1] + cor_stat_neg[i+4]) * 1. / (sum_stat_neg[i] + sum_stat_neg[i+1] + sum_stat_neg[i+4]))

    return result_4

def contrast(labels, but_total, preds):
    i = 0
    
    but_sum = 0
    but_cor = 0
    but_tot = 0
    for pred in preds:
        but = but_total[i]
        label = labels[i]
        
        if len(but) != 0:
            but_tot += 1
            if but[0] >= 0:
                but_sum += 1
                if label[but[0]] == pred[but[0]] and label[but[1]] == pred[but[1]] and label[but[2]] == pred[but[2]]:
                    but_cor += 1
        i += 1
        
    print(but_sum)
        
    return but_cor * 1. / but_sum

def simple_accuracy_phrase(preds, labels, task_name):
    tot = 0
    i = 0
    s = 0
    tot_sen = 0
    cor_sen = 0
    
    swap_total = np.load("swap_test.npy")
    edge_total = np.load("edge_test.npy")
    edge_swap_total = np.load("edge_swap_test.npy")
    but_total = np.load("but_new.npy")
    neg_total = np.load("neg_new.npy")
    
    cor_stat = [0] * 128
    sum_stat = [0] * 128
    cor_stat_neg = [0] * 128
    sum_stat_neg = [0] * 128
    
    for pred in preds:
        tot_sen += 1
        for j in range(len(pred)):
            if labels[i][j] != -1:
                if j == 0:
                    tot += 1
                    if pred[j] == labels[i][j]:
                        cor_sen += 1
                        s += 1
                else:
                    tot += 1
                    if pred[j] == labels[i][j]:
                        s += 1
                
                if task_name == "sstphrase":
                    if str(pred[j]) == '0' or str(pred[j]) == '1':
                        preds[i][j] = 0
                    if str(pred[j]) == '3' or str(pred[j]) == '4':
                        preds[i][j] = 1
                    if str(pred[j]) == '2':
                        preds[i][j] = 2
                    if str(labels[i][j]) == '0' or str(labels[i][j]) == '1':
                        labels[i][j] = 0
                    if str(labels[i][j]) == '3' or str(labels[i][j]) == '4':
                        labels[i][j] = 1
                    if str(labels[i][j]) == '2':
                        labels[i][j] = 2
                    
        i += 1
    
    if task_name == "sstphrase":            
        i = 0
        for pred in preds:
            sum_cor = 0
            label = labels[i]
            tot_tmp = 0
            for j in range(len(pred)):
                if label[j] != -1:
                    tot_tmp += 1
                    if pred[j] == label[j]:
                        sum_cor += 1
                        
            cor_stat[swap_total[i]] += sum_cor
            sum_stat[swap_total[i]] += tot_tmp
            
            cor_stat_neg[neg_total[i]] += sum_cor
            sum_stat_neg[neg_total[i]] += tot_tmp
            i += 1
            
        result_0 = s / tot * 1.0
        result_1 = cor_sen * 1.0 / tot_sen
        result_2 = global_swap(sum_stat, cor_stat)
        result_3 = local_swap(preds, labels, edge_total, edge_swap_total)
        result_4 = negation(cor_stat_neg, sum_stat_neg)
        result_5 = contrast(labels, but_total, preds)
        
        return [result_0, result_1, result_2, result_3, result_4, result_5]
    else:
        return cor_sen * 1.0 / tot_sen
        

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def avg_recall(preds, labels):
    avg_r = 0.0
    corr = [0, 0, 0]
    tot = [0, 0, 0]
    for i in range(len(labels)):
        tot[int(labels[i])] += 1
        if str(preds[i]) == str(labels[i]):
            corr[int(preds[i])] += 1
            
    avg_r = 1./3 * corr[0] / tot[0] + 1./3 * corr[1] / tot[1] + 1./3 * corr[2] / tot[2]
    return avg_r

def micro_f1(preds, labels):
    preds_set = []
    label_set = []
    
    corr = 0
    tot = 0
    
    for i in range(len(labels)):
        if preds[i] != 3:
            tot += 1
        if int(labels[i]) != 3:
            preds_set.append(int(preds[i]))
            label_set.append(int(labels[i]))
            if labels[i] == preds[i]:
                corr += 1
                
    p = corr * 1. / len(preds_set)
    r = corr * 1. / tot
            
    f1 = 2 * p * r / (p+r)
    return f1

def pearson_and_spearman(preds, labels):                      
    pearson_corr = pearsonr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
    }


def compute_metrics(task_name, preds, labels, preds_ans=None):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sst-3":
        return {"acc": simple_accuracy_phrase(preds, labels, task_name)}
    elif task_name == "sstphrase":
        result = simple_accuracy_phrase(preds, labels, task_name)
        return {"phrase_acc": result[0], "sst-5_acc": result[1], "global_swap_acc": result[2], "local_swap_acc": result[3], "neg_acc": result[4], "but_acc": result[5]}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "twitter":
        return {"acc": avg_recall(preds, labels)}
    elif task_name == "emocontext":
        return {"acc": micro_f1(preds, labels)}
    elif task_name == "emoint":
        return pearson_and_spearman(preds, labels)
    else:
        raise KeyError(task_name)

processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sst-3": Sst3Processor,
    "sstphrase": SstPhraseProcessor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "twitter": TwitterProcessor,
    "emocontext": EmoContextProcessor,
    "emoint": EmoIntProcessor
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sst-3": "classification",
    "sstphrase": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "twitter": "classification",
    "emocontext": "classification",
    "emoint": "classification"
}
