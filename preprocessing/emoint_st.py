# coding=utf-8

from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import json
import pickle as pkl
import argparse
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def toTree(expression):
    count = 0
    tree = dict()
    express = dict()
    msg = ""
    stack = list()
    for char in expression:
        if char == '(':
            count += 1
            if msg in phrase_dict.keys():
                phrase_dict[msg] += 1
                if phrase_dict[msg] % 2 == 1:
                    stack.append(msg + '_' + str(phrase_dict[msg] / 2))
                else:
                    if phrase_dict[msg] != 2:
                        stack.append(msg + '_' + str((phrase_dict[msg] - 1) / 2))
                    else:
                        stack.append(msg)
            else:
                phrase_dict[msg] = 1
                stack.append(msg)
            msg = ""
        elif char == ')':
            if len(stack) == 0:
                continue
            parent = stack.pop()

            if ' ' in msg:
                msg = msg[msg.find(' ') + 1:]
                express[msg] = msg

            if parent not in tree:
                tree[parent] = list()
            
            if msg in whole_dict.keys():
                whole_dict[msg] += 1
                tree[parent].append(msg + ' ' + str(whole_dict[msg] - 1))
                whole_dict[msg + ' ' + str(whole_dict[msg] - 1)] = 1
            else:
                whole_dict[msg] = 1
                tree[parent].append(msg)

            if parent == '':
                continue
            if parent not in express.keys():
                if msg != '':
                    express[parent] = express[msg]
                else:
                    express[parent] = ''
            else:
                express[parent] += ' ' + express[msg]

            msg = parent
        else:
            msg += char 
            
    return tree, express

parser = argparse.ArgumentParser()
parser.add_argument("--stage",
                    default="train",
                    type=str,
                    required=True,
                    help="train or test or dev")
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir.")
parser.add_argument("--tree_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input tree dir.")
parser.add_argument("--domain",
                    default=None,
                    type=str,
                    required="joy",
                    help="joy or fear or sad or anger")
args = parser.parse_args()

with open(os.path.join(args.tree_dir, 'emoint_'+args.stage+'_'+args.domain+'.txt.json'), 'r') as f:
    data = json.load(f)
    
f1 = open(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'_text_new.txt'), 'w')
f2 = open(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'.txt'), 'r')

idx = 0
sen_list = []
for line in f2:
    sen = line.strip()
    sen_list.append(sen)
    if ':(' in sen:
        data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace(':(', 'anger')
    if ':-)' in sen:
        data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace(':-)', 'happy')
    if '((:' in sen:
        data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace('anger((:','anger')
    if '(#euthanasia' in sen:
        data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace('(#euthanasia','#euthanasia')
    words = sen.split(' ')
    for word in words:
        if '(' in word:
            data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace(word, word.replace('(', ''))
        word = word.replace('(', '')
        if ')' == word:
            data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace("(X|sentiment=3|prob=0.990 )", "(X|sentiment=3|prob=0.990 .")
        else:
            if ')' in word:
                data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace(word, word.replace(')', ''))
                data["sentences"][idx]["sentimentTree"] = data["sentences"][idx]["sentimentTree"].replace("U19angerODI):", "U19angerODI:")
    
    idx += 1
    
emoint_label = np.load(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'_label.npy'))
expression = []  
sen_set = data["sentences"]

for sen in sen_set:
    if sen["sentimentTree"] != "(ROOT|sentiment=2|prob=0.915 (VP|sentiment=2|prob=0.983 1) (.|sentiment=2|prob=0.997 .))":
        expression.append(sen["sentimentTree"].replace("(Reuters", "Reuters").replace(":(", "anger").replace("(via", "via"))
    else:
        expression.append("split")

max_len = 128
maxi = 0
s = 0
split = []
span = []
span_3 = []
label = []
for i in range(len(expression)):
    if i == 27211:
        split.append(1)
        continue
    if expression[i] == "split":
        split.append(1)
        continue
    else:
        split.append(0)
    
    phrase_dict = dict()
    whole_dict = dict()
    word_dict = dict()
    idx_dict = dict()
    sentiment_label = []
    
    expression[i] = expression[i].replace('\n','')
    expression[i] = expression[i].replace('\r','')
    expression[i] = expression[i].replace('  ','')
    expression[i] = expression[i].replace(' (','(')
    tree, express = toTree(expression[i])
    
    word_set = []
    tot = 0
    for key in whole_dict.keys():
        if key not in phrase_dict.keys():
            key_tmp = key + '&_'
            s_str = key_tmp[key_tmp.find(' ')+1:]
            s_str = s_str[:s_str.find('&_')]
            word_set.append(s_str)
            idx_dict[s_str] = tot
            tot += 1
    
    tot_word = tot
    span_mask = []
    span_mask_3 = []
    sen_key = ""
    flag = 0
    for key in phrase_dict.keys():
        if key == '':
            continue
        sen_key = key
            
        if sen_key not in express.keys():
            span_mask_item = [1] * (max_len - 1)
            span_mask_item.append(0)
            span_mask_pad = [0] * max_len
                
            span_mask.append(span_mask_item)
            span_mask_3.append(span_mask_item + [0] * max_len)
            for k in range(max_len-1):
                span_mask.append(span_mask_pad)
                span_mask_3.append(span_mask_pad + [0] * max_len)
            
            f1.write(sen_list[i]+'\n')
            span.append(span_mask)
            label.append(emoint_label[i])
            span_3.append(span_mask_3)
            flag = 1
            break
    
    if flag == 1:
        split[-1] = 1
        continue
        
    if tot > max_len - 2:
        split[-1] = 1
        continue
    
    span_token = []
    count = 0
    label2phrase = []
    for key in phrase_dict.keys():
        if key == '':
            continue
        if count == 0:
            sen_str = express[key]
        span_token.append((tokenizer.tokenize(express[key]), key))
        count += 1
        idx_dict[key] = tot
        label2phrase.append(key)
        tot += 1
        
    if len(span_token) == 0:
        split[-1] = 1
        continue 
        
    sen_token = span_token[0][0]
    
    if len(sen_token) > max_len - 2:
        split[-1] = 1
        continue
    
    if "*** / / / ." in sen_str:
        split[-1] = 1
        continue
     
    span_info = []
    for j in range(max_len):
        if j < len(span_token):
            span_mask_item = []
            for start in range(len(sen_token)-len(span_token[j][0])+1):
                if sen_token[start:start+len(span_token[j][0])] == span_token[j][0]:
                    span_mask_item += [0] * start
                    span_mask_item += [1] * len(span_token[j][0])
                    span_mask_item += [0] * (max_len - 1 - len(span_mask_item))
                    span_mask_item_real = span_mask_item + [0] * (max_len - len(span_mask_item))
                    span_info.append((span_mask_item, span_token[j][1]))
                    
                    break
        else:
            span_mask_item = [0] * (max_len - 1)
            span_info.append((span_mask_item, ""))
            span_mask_item_real = [0] * max_len
        
        span_mask.append(span_mask_item_real)
    
    if tot > max_len:
        split[-1] = 1
        continue
    
    f1.write(sen_str+'\n')
    
    label.append(emoint_label[i])
    span.append(span_mask)
    span_mask_3 = []
    
    for key in label2phrase:
        if key == '':
            continue
        son_list = tree[key]
        par_span = []
        son_span = []
        son_index = []
        par_index = idx_dict[key]
        for son in son_list:
            if son == '':
                continue
            if " " in son:
                son_tmp = son + '&_'
                s_str = son_tmp[son_tmp.find(' ')+1:]
                son = s_str[:s_str.find('&_')]
            
            for info in span_info:
                if key == info[1]:
                    par_span = info[0]
                    break
            
            for info in span_info:
                if son == info[1]:
                    son_index.append(idx_dict[son] - tot_word)
                    son_span.append(info[0])
                    
        if len(son_span) == 1:
            son_span_tmp = []
            for t in range(len(par_span)):
                if t >= len(son_span[0]) and par_span[t] == 1:
                    son_span_tmp.append(1)
                    continue
                if par_span[t] == 1 and son_span[0][t] == 0:
                    son_span_tmp.append(1)
                else:
                    son_span_tmp.append(0)
            son_span.append(son_span_tmp)
            
        span_3_tmp = []
        span_extended = [0] * max_len
        span_extended_1 = [0] * max_len
        span_extended_2 = [0] * max_len
        for t in range(len(son_index)):
            span_extended[son_index[t]] = 1
        if len(son_index) == 2:
            span_extended_1[son_index[0]] = 1
            span_extended_2[son_index[1]] = 1
        if len(son_index) == 0:
            if len(par_span) == 0:
                span_3_tmp = [0] * (max_len - 1)
            else:
                for t in range(max_len - 1):
                    span_3_tmp.append(par_span[t])
                    
            for t in range(max_len):
                if t == par_index - tot_word:
                    span_3_tmp.append(1)
                else:
                    span_3_tmp.append(span_extended[t])
            span_3_tmp.append(0)
            
        if len(son_index) == 1:
            if len(son_span[1]) == 0:
                span_3_tmp = [0] * (max_len - 1)
            else:
                for t in range(max_len - 1):
                    span_3_tmp.append(son_span[1][t])
                    
            for t in range(max_len):
                if t == par_index - tot_word:
                    span_3_tmp.append(1)
                else:
                    span_3_tmp.append(span_extended[t])
            span_3_tmp.append(0)
                
        if len(son_index) == 2:
            span_3_tmp = [0] * (max_len - 1)
            for t in range(max_len):
                if t == par_index - tot_word:
                    span_3_tmp.append(1)
                else:
                    span_3_tmp.append(span_extended[t])
            span_3_tmp.append(0)
        
        assert len(span_3_tmp) == max_len * 2
        span_mask_3.append(span_3_tmp)
        
    for t in range(max_len - len(span_mask_3)):
        span_mask_3.append([0] * max_len * 2)
        
    s += 1
    span_3.append(span_mask_3)

# print(len(span), len(label))
np.save(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'_span.npy'), span)
np.save(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'_span_3.npy'), span_3)
np.save(os.path.join(args.data_dir, 'emoint_label_'+args.stage+'_'+args.domain+'.npy'), label)
np.save(os.path.join(args.data_dir, 'emoint_'+args.stage+'_'+args.domain+'_split.npy'), split)
    
f2.close()
f1.close()
f.close()