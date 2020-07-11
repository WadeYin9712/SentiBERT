# coding=utf-8
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import pickle as pkl
import argparse
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def toTree(expression): 
    count = 0
    tree = dict()
    express = dict()
    sentiment = dict()
    msg = "" 
    stack = list() 
    for char in expression: 
        if char == '(':
            count += 1
            # print(msg)
            if msg in phrase_dict.keys():
                phrase_dict[msg] += 1
                if phrase_dict[msg] % 2 == 1:
                    stack.append(msg+'_'+str(phrase_dict[msg]/2))
                else:
                    if phrase_dict[msg] != 2:
                        stack.append(msg+'_'+str((phrase_dict[msg]-1)/2))
                    else:
                        stack.append(msg)
            else:
                phrase_dict[msg] = 1
                stack.append(msg)
            msg = "" 
        elif char == ')':
            parent = stack.pop()
                
            if ' ' in msg:
                msg_tmp = msg
                msg = msg[msg.find(' ')+1:]
                express[msg] = msg
                sentiment[msg] = int(msg_tmp[msg_tmp.rfind("/*/*=")-1])
                
            if parent not in tree:
                tree[parent] = list()
            #print(msg)
            if msg in whole_dict.keys():
                whole_dict[msg] += 1
                # if " " in msg:
                tree[parent].append(msg+' '+str(whole_dict[msg]-1))
                whole_dict[msg+' '+str(whole_dict[msg]-1)] = 1
            else:
                whole_dict[msg] = 1
                tree[parent].append(msg)
                
            if parent == '':
                continue
            sentiment[parent] = int(parent[parent.rfind("/*/*=")+5])
            if parent not in express.keys():
                express[parent] = express[msg]
            else:
                express[parent] += ' ' + express[msg]
                
            msg = parent 
        else: 
            msg += char 
            
    return tree, express, sentiment

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
args = parser.parse_args()

data_dir = args.data_dir
tree_dir = args.tree_dir
f1 = open(os.path.join(tree_dir, "sstphrase_"+args.stage+".txt"), 'r')
f2 = open(os.path.join(data_dir, "sstphrase_"+args.stage+"_text_new.txt"), 'w')

max_len = 128
but_sum = 0
maxi = 0
s = 0
sum1 = 0
split = []
sentiment_total = []
graph = np.zeros((10000, max_len*2, max_len*2))
i = 0
span = [] # all tokens
span_3 = [] # phrase A + phrase B
swap = []
edge = []
but = []
neg = []
edge_swap = []
diff_sen = [0] * 3
senti_ans = {}
for line in f1:
    line = line.strip()
    
    new_line = ""
    char_count = 0
    node_count = 0
    for char in line:
        if char == ' ' and line[char_count+1] == '(':
            char_count += 1
            continue
        else:
            if char >= '0' and char <= '4' and line[char_count-1] == '(':
                new_line += str(node_count) + "/*/*="
                node_count += 1
            new_line += char
        char_count += 1
            
    line = new_line
    split.append(0)
    
    phrase_dict = dict()
    whole_dict = dict()
    word_dict = dict()
    idx_dict = dict()
    
    expression = line.replace('\n','')
    expression = expression.replace('\r','')
    expression = expression.replace(' (','(')

    tree, express, sentiment = toTree(expression)
    
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
    if tot > max_len - 2:
        split[-1] = 1
        span.append([-1])
        span_3.append([-1])
        swap.append([-1])
        neg.append([-1])
        edge_swap.append([-1])
        but.append([-1, -1])
        edge.append([[-1, -1, -1]])
        sentiment_total.append([-1])
        f2.write('\n')
        i += 1
        
        continue
    
    sentiment_label = []
    span_token = []
    count = 0
    label2phrase = []
    for key in phrase_dict.keys():
        if key == '':
            continue
        if count == 0:
            sen_str = express[key]
        sentiment_label.append(sentiment[key])
        span_token.append((tokenizer.tokenize(express[key]), key))
        count += 1
        idx_dict[key] = tot
        label2phrase.append(key)
        tot += 1
        
    neg_num = sen_str.count(" not") + sen_str.count("n't") + sen_str.count(" no ")
    sen_token = span_token[0][0]
    
    if len(sen_token) > max_len - 2:
        split[-1] = 1
        span.append([-1])
        span_3.append([-1])
        swap.append([-1])
        neg.append([-1])
        edge_swap.append([-1])
        edge.append([[-1, -1, -1]])
        but.append([-1, -1])
        sentiment_total.append([-1])
        f2.write('\n')
        i += 1
        continue
    
    span_mask = [] 
    span_info = []
    for j in range(max_len):
        if j < len(span_token):
            span_mask_item = []
            for start in range(len(sen_token)-len(span_token[j][0])+1):
                if sen_token[start:start+len(span_token[j][0])] == span_token[j][0]:
                    span_mask_item += [0] * start
                    span_mask_item += [1] * len(span_token[j][0])
                    span_mask_item += [0] * (max_len - 1 - len(span_mask_item))
                    span_mask_item_real = span_mask_item + [0] * (max_len * 2 - len(span_mask_item))
                    span_info.append((span_mask_item, span_token[j][1]))
                    break
        else:
            span_mask_item = [0] * (max_len - 1)
            span_info.append((span_mask_item, ""))
            span_mask_item_real = [0] * max_len * 2
        span_mask.append(span_mask_item_real)
    
    assert len(sentiment_label) == len(span_token)
    
    if tot != tot_word * 2 - 1:
        split[-1] = 1
        span.append([-1])
        span_3.append([-1])
        swap.append([-1])
        neg.append([-1])
        edge.append([[-1, -1, -1]])
        but.append([-1, -1])
        edge_swap.append([-1])
        sentiment_total.append([-1])
        f2.write('\n')
        i += 1
        continue
    
    but_sen = []
    for key in label2phrase:
        if key == '':
            continue
        son_list = tree[key]
        
        flag = False
        idx_key = -1
        x_key = -1
        if son_list[0] == "but" or son_list[1] == "but":
            for son in son_list:
                if son in label2phrase:
                    if "but" in express[key] and "but" not in express[son]:
                        flag = True
                        idx_key = idx_dict[key]
                        x_key = son
                        break
            
        if flag:
            y_key = -1
            for key_par in label2phrase:
                if key_par == '':
                    continue
                son_par = tree[key_par]
                idx_par_key = idx_dict[key_par]
                
                son_idx = []
                flag_1 = False
                for son_son in son_par:
                    if son_son in label2phrase:
                        if idx_dict[son_son] == idx_key:
                            flag_1 = True
                        son_idx.append(son_son)
                    
                if len(son_idx) == 2 and flag_1:
                    if idx_dict[son_idx[0]] == idx_key:
                        y_key = son_idx[1]
                    if idx_dict[son_idx[1]] == idx_key:
                        y_key = son_idx[0]
                    if sentiment[x_key] == 0 or sentiment[x_key] == 1:
                        sen_x = 0
                    if sentiment[x_key] == 3 or sentiment[x_key] == 4:
                        sen_x = 1
                    if sentiment[x_key] == 2:
                        sen_x = 2
                        
                    if sentiment[y_key] == 0 or sentiment[y_key] == 1:
                        sen_y = 0
                    if sentiment[y_key] == 3 or sentiment[y_key] == 4:
                        sen_y = 1
                    if sentiment[y_key] == 2:
                        sen_y = 2  
                    if sen_x != sen_y and y_key in label2phrase:
                        but_sen= [idx_dict[x_key]-tot_word, idx_dict[y_key]-tot_word, idx_dict[key_par]-tot_word]
                        but_sum += 1
                        
    span.append(span_mask)
    
    span_mask_1 = []
    span_mask_2 = []
    span_mask_3 = []
    edge_tot = []
    edge_swap_stat = []
    
    diff = 0
    for key in label2phrase:
        if key == '':
            continue
        son_list = tree[key]
        par_span = []
        son_span = []
        son_index = []
        edge_info = []
        par_index = idx_dict[key]
        edge_info.append(idx_dict[key]-tot_word)
        
        diff_node = 0
        
        for son in son_list:
            if " " in son:
                son_tmp = son + '&_'
                s_str = son_tmp[son_tmp.find(' ')+1:]
                son = s_str[:s_str.find('&_')]
                
            if son in label2phrase:
                if sentiment[key] == 0 or sentiment[key] == 1:
                    sen_key = 0
                if sentiment[key] == 3 or sentiment[key] == 4:
                    sen_key = 1
                if sentiment[key] == 2:
                    sen_key = 2
                    
                if sentiment[son] == 0 or sentiment[son] == 1:
                    sen_son = 0
                if sentiment[son] == 3 or sentiment[son] == 4:
                    sen_son = 1
                if sentiment[son] == 2:
                    sen_son = 2    
                    
                if sen_key != sen_son:
                    diff += 1
                    diff_node += 1
                    
                edge_info.append(idx_dict[son]-tot_word)
            
            for info in span_info:
                if key == info[1]:
                    par_span = info[0]
                    break
            
            for info in span_info:
                if son == info[1]:
                    son_index.append(idx_dict[son] - tot_word)
                    son_span.append(info[0])
        
        edge_info += [-1] * (3 - len(edge_info))
        edge_tot.append(edge_info)
        edge_swap_stat.append(diff_node)
        
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
            for t in range(max_len - 1):
                span_3_tmp.append(par_span[t])
            for t in range(max_len):
                if t == par_index - tot_word:
                    span_3_tmp.append(1)
                else:
                    span_3_tmp.append(span_extended[t])
            span_3_tmp.append(0)
            
        if len(son_index) == 1:
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
        
    sentiment_total.append(sentiment_label)
    s += 1
    i += 1
    
    senti_ans[sen_str.lower()+' '] = sentiment_label
    swap.append(diff)
    edge.append(edge_tot)
    edge_swap.append(edge_swap_stat)
    but.append(but_sen)
    neg.append(neg_num)
    f2.write(sen_str+'\n')
    # print(sen_str)
    span_3.append(span_mask_3)    
    
# print(but_sum)  
print(len(span), len(span_3), len(split), len(sentiment_total))

np.save(os.path.join(data_dir, 'sstphrase_'+args.stage+'_span.npy'), span)
np.save(os.path.join(data_dir, 'sstphrase_'+args.stage+'_span_3.npy'), span_3)
if args.stage == "test":
    np.save(os.path.join(data_dir, 'swap_'+args.stage+'_new.npy'), swap)
    np.save(os.path.join(data_dir, 'edge_'+args.stage+'_new.npy'), edge)
    np.save(os.path.join(data_dir, 'edge_swap_'+args.stage+'_new.npy'), edge_swap)
    np.save(os.path.join(data_dir, 'but_new.npy'), but)
    np.save(os.path.join(data_dir, 'neg_new.npy'), neg)

np.save(os.path.join(data_dir, 'sstphrase_split_'+args.stage+'.npy'), split)
np.save(os.path.join(data_dir, 'sstphrase_label_'+args.stage+'.npy'), sentiment_total)
    
f1.close()
f2.close()