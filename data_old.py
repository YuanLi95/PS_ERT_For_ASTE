import math

import torch
import numpy as np
import  pickle
import os
import json
from torch.nn import  functional as F
sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
from transformers import BertTokenizer


def get_spans(tags):
    '''for BIO tag'''
    tags = tags.strip().split()
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


def get_evaluate_spans(tags, length, token_range):
    '''for BIO tag'''
    spans = []
    start = -1
    for i in range(length):
        l, r = token_range[i]
        if tags[l] == -1:
            continue
        elif tags[l] == 1:
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[l] == 0:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    def __init__(self, tokenizer, sentence_pack,dependency_mask_pack,edge_data_pack,
                 syntax_position_pack,adj_index_undir,position_tokenizer,dependency_tokenizer, args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.tokens = self.sentence.strip().split()
        self.sen_length = len(self.tokens)
        self.token_range = []
        # print(self.sentence)
        # exit(<?)
        self.bert_tokens = tokenizer.encode(self.sentence)
        # print(self.bert_tokens)
        # print(self.sentence)
        self.length = len(self.bert_tokens)
        self.dependency_mask_seq = torch.zeros(self.length, self.length).long()
        self.edge_data_seq = torch.zeros(self.length, self.length).long()
        self.syntax_position_seq = torch.zeros(self.length, self.length).long()



        position_matrix = position_tokenizer.position_to_index(syntax_position_pack)
        # print(self.bert_tokens)
        # print(self.length)

        dependency_edge, adj_dependency_index,adj_dependency_mask = dependency_tokenizer.dependency_to_index(edge_data_pack, dependency_mask_pack)
        # print(dependency_edge)
        # exit( )

        # adj_index_undir =



        self.dependency_mask_seq[0:self.length, 0:self.length] = torch.from_numpy(dependency_mask_pack)
        self.syntax_position_seq[0:self.length, 0:self.length] = torch.from_numpy(position_matrix)
        self.edge_data_seq[0:self.length, 0:self.length] = torch.from_numpy(dependency_edge)


        self.length = len(self.bert_tokens)
        self.bert_tokens_padding = torch.zeros(self.length).long()
        self.aspect_tags = torch.zeros(self.length).long()
        self.opinion_tags = torch.zeros(self.length).long()
        self.tags = torch.zeros(self.length, self.length).long()
        self.mask = torch.zeros(self.length)
        self.adj_index_undir =adj_index_undir
        self.adj_dependency_index = adj_dependency_index
        self.adj_dependency_mask =adj_dependency_mask

        for i in range(self.length):
            self.bert_tokens_padding[i] = self.bert_tokens[i]
        self.mask[:self.length] = 1

        token_start = 1
        # print(self.tokens)
        token_list = []
        for i, w, in enumerate(self.tokens):
            token_list.append(tokenizer.tokenize(w))
            token_end = token_start + len(tokenizer.encode(w, add_special_tokens=False))

            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        # print(token_list)
        assert self.length == self.token_range[-1][-1]+2

        self.aspect_tags[self.length:] = -1
        self.aspect_tags[0] = -1
        self.aspect_tags[self.length-1] = -1

        self.opinion_tags[self.length:] = -1
        self.opinion_tags[0] = -1
        self.opinion_tags[self.length - 1] = -1

        self.tags[:, :] = -1
        for i in range(1, self.length-1):
            for j in range(i, self.length-1):
                self.tags[i][j] = 0
        # print(self.token_range)


        for triple in sentence_pack['triples']:

            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            # print(aspect)
            # print(opinion)
            aspect_span = get_spans(aspect)
            # print(aspect_span)

            opinion_span = get_spans(opinion)
            # print(opinion_span)
            # print(triple['sentiment'])
            # print(sentiment2id[triple['sentiment']])

            '''set tag for aspect'''
            for l, r in aspect_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = 1
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    al, ar = self.token_range[i]
                    self.aspect_tags[al] = set_tag
                    self.aspect_tags[al+1:ar+1] = -1
                    '''mask positions of sub words'''
                    self.tags[al+1:ar+1, :] = -1
                    self.tags[:, al+1:ar+1] = -1
            # print(self.tags)
            '''set tag for opinion'''
            for l, r in opinion_span:
                start = self.token_range[l][0]
                end = self.token_range[r][1]
                for i in range(start, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = 2
                for i in range(l, r+1):
                    set_tag = 1 if i == l else 2
                    pl, pr = self.token_range[i]
                    self.opinion_tags[pl] = set_tag
                    self.opinion_tags[pl+1:pr+1] = -1
                    self.tags[pl+1:pr+1, :] = -1
                    self.tags[:, pl+1:pr+1] = -1

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            sal, sar = self.token_range[i]
                            spl, spr = self.token_range[j]
                            self.tags[sal:sar+1, spl:spr+1] = -1
                            if args.task == 'pair':
                                # print("11111111111111222")
                                # exit()
                                if i > j:
                                    self.tags[spl][sal] = 3
                                else:
                                    self.tags[sal][spl] = 3
                            elif args.task == 'triplet':
                                if i > j:
                                    self.tags[spl][sal] = sentiment2id[triple['sentiment']]
                                else:
                                    # print( self.tags[sal][spl] )
                                    # exit()
                                    self.tags[sal][spl] = sentiment2id[triple['sentiment']]
            # for i in range(self.tags.shape[0]):
            # #     print(self.tags[16][:])
            # #     print(self.tags[17][:])
            #     print(i, self.tags[i][:])
            # exit()



def load_data_instances(path, position_tokenizer,dependency_tokenizer, args):
    nstances = list()
    sentence_packs = json.load(open(path))
    fout_undir_file = os.path.join("%sundirbert.graph" % path)
    syntax_position_file = os.path.join("%sbert.syntaxPosition" % path)
    dependency_type_file = os.path.join("%sbert.dependency" % path)
    adj_index_undir_file = os.path.join("%sbert.aj_undir_index" % path)
    dependency_undir = open(fout_undir_file, 'rb')
    edge_type = open(dependency_type_file, "rb")
    syntax_position = open(syntax_position_file, "rb")
    adj_index_undir = open(adj_index_undir_file,"rb")

    dependency_mask_data = pickle.load(dependency_undir)
    edge_data = pickle.load(edge_type)
    syntax_position_data = pickle.load(syntax_position)
    adj_index = pickle.load(adj_index_undir)


    #
    dependency_undir.close()
    edge_type.close()
    syntax_position.close()
    instances = list()
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)

    for i, sentence_pack in enumerate(sentence_packs):
        instances.append(Instance(tokenizer, sentence_pack, dependency_mask_data[i], edge_data[i],
                                  syntax_position_data[i],adj_index[i],position_tokenizer,
                                  dependency_tokenizer, args))

    return instances


class DataIterator(object):
    def __init__(self, instances, args):
        self.instances = instances
        self.args = args
        self.batch_count = math.ceil(len(instances)/args.batch_size)

    def get_batch(self, index):
        sentence_ids = []
        sentences = []
        sens_lens = []
        token_ranges = []
        bert_tokens = []
        lengths = []
        masks = []
        aspect_tags = []
        opinion_tags = []
        tags = []
        adj_dependency_index_all = []

        adj_dependency_mask_all = []

        adj_index_undir_all = []

        dependency_masks = []
        syntactic_position_datas = []
        edge_datas = []
        max_token_length=0
        max_dependency_length=0
        batch_dependency_length=[]

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):
            now_line_max_token_length =self.instances[i].length
            #这里是一个二维度的东西 每一个元素不等长

            now_dependency_max_length = max([ len(i) for i in self.instances[i].adj_dependency_index])
            if max_dependency_length<=now_dependency_max_length:
                max_dependency_length= now_dependency_max_length
            if max_token_length<= now_line_max_token_length:
                max_token_length = now_line_max_token_length

        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.instances))):

            text_length = self.instances[i].length
            text_padding = [0]*(max_token_length-text_length)
            line_tags = [-1]*(max_token_length-text_length)


            # 找到每一个需要填充多少个adj_dependency_index和adj_index_undir
            now_dependency_list=[]
            for index in range(len(self.instances[i].adj_dependency_index)):
                now_dependency_list.append(len(self.instances[i].adj_dependency_index[index]))
                self.instances[i].adj_dependency_index[index]+=[0]*(max_dependency_length-len(self.instances[i].adj_dependency_index[index]))
                #这里的1代表unk类型在star-dependency_transformer中需要pad两个元素
                self.instances[i].adj_dependency_index[index]+=[1]*1

                self.instances[i].adj_index_undir[index]+=[0]*(max_dependency_length-len(self.instances[i].adj_index_undir[index]))
                self.instances[i].adj_dependency_mask[index]+=[0]*(max_dependency_length-len(self.instances[i].adj_dependency_mask[index]))
                # 这里的1代表在star-dependency_transformer中需要pad两个元素
                self.instances[i].adj_dependency_mask[index]+=[1]*1

                # print(len(self.instances[i].adj_index_undir[index]))
                # print(len(self.instances[i].adj_dependency_index[index]))

            batch_dependency_length.append(F.pad(torch.tensor(now_dependency_list),(0,max_token_length-text_length),'constant',value=0))
            sentence_ids.append(self.instances[i].id)
            sentences.append(self.instances[i].sentence)
            sens_lens.append(self.instances[i].sen_length)
            token_ranges.append(self.instances[i].token_range)
            lengths.append(self.instances[i].length)
            bert_tokens.append(torch.cat((self.instances[i].bert_tokens_padding ,torch.LongTensor(text_padding)),axis=0))

            masks.append(torch.cat((self.instances[i].mask,torch.FloatTensor(text_padding)),axis=0).type(torch.LongTensor))
            aspect_tags.append(torch.cat((self.instances[i].aspect_tags,torch.tensor(line_tags)),axis=0))
            opinion_tags.append(torch.cat((self.instances[i].opinion_tags,torch.tensor(line_tags)),axis=0))
            tags.append(F.pad(self.instances[i].tags,(0,max_token_length-text_length,0,max_token_length-text_length),'constant',value=-1))
            # print(self.instances[i].adj_index_undir.shape)
            # exit()
            adj_index_undir_all.append(F.pad(torch.tensor(self.instances[i].adj_index_undir),(0,0,0,max_token_length-text_length),'constant',value=0))
            # print(torch.tensor(self.instances[i].adj_dependency_index).shape)
            # print(torch.tensor(self.instances[i].adj_index_undir).shape)
            # print(len(bert_tokens))
            adj_dependency_index_all.append(F.pad(torch.tensor(self.instances[i].adj_dependency_index),(0,0,0,max_token_length-text_length),'constant',value=0))
            adj_dependency_mask_all.append(F.pad(torch.tensor(self.instances[i].adj_dependency_mask),(0,0,0,max_token_length-text_length),'constant',value=0))


            dependency_masks.append(F.pad(self.instances[i].dependency_mask_seq,(0,max_token_length-text_length,0,max_token_length-text_length),'constant',value=0))

            syntactic_position_datas.append(F.pad(self.instances[i].syntax_position_seq,(0,max_token_length-text_length,0,max_token_length-text_length),'constant',value=0))

            edge_datas.append(F.pad( self.instances[i].edge_data_seq,(0,max_token_length-text_length,0,max_token_length-text_length),'constant',value=0))


        # sentence_ids= torch.stack(sentence_ids).to(self.args.device)

        bert_tokens = torch.stack(bert_tokens).long().to(self.args.device)
        lengths = torch.tensor(lengths).long().to(self.args.device)
        masks = torch.stack(masks).long().to(self.args.device)
        aspect_tags = torch.stack(aspect_tags).long().to(self.args.device)
        opinion_tags = torch.stack(opinion_tags).long().to(self.args.device)
        # token_ranges = torch.stack(token_ranges).to(self.args.device)
        tags = torch.stack(tags).long().to(self.args.device)
        adj_index_undir_all = torch.stack(adj_index_undir_all).long().to(self.args.device)

        adj_dependency_index_all=torch.stack(adj_dependency_index_all).long().to(self.args.device)
        batch_dependency_length = torch.stack(batch_dependency_length).long().to(self.args.device)




        dependency_masks = torch.stack(dependency_masks).long().to(self.args.device)
        syntactic_position_datas = torch.stack(syntactic_position_datas).long().to(self.args.device)
        edge_datas = torch.stack(edge_datas).long().to(self.args.device)
        adj_dependency_mask_all = torch.stack(adj_dependency_mask_all).long().to(self.args.device)
        # print(batch_dependency_length)
        # print(adj_dependency_mask_all.shape)
        # exit()


        # print(len(sentence_ids))
        # print(bert_tokens.shape)
        # print(lengths.shape)
        # print(len(sens_lens))
        # print(masks.shape)
        # print(len(token_ranges))
        # print(dependency_masks.shape)
        # print(syntactic_position_datas.shape)
        # print(edge_datas.shape)
        # print(adj_index_undir_all.shape)
        # print(adj_dependency_index_all.shape)
        # print(aspect_tags.shape)
        # print(opinion_tags.shape)
        # print(tags.shape)
        # exit()

        return sentence_ids, bert_tokens, lengths, masks, sens_lens, token_ranges,  \
               syntactic_position_datas, adj_index_undir_all,adj_dependency_index_all,adj_dependency_mask_all,batch_dependency_length, aspect_tags, opinion_tags, tags,dependency_masks
