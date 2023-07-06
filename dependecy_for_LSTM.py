# -*- coding: utf-8 -*-

import numpy as np
import pickle
import spacy
import networkx as nx
import  re
from spacy import  displacy
from pytorch_pretrained_bert import  BertTokenizer

# depenpency_type = "spacy"
import json
token_nize_for_tokenize = spacy.load('en_core_web_trf')
def tokenize(text):
    text=text.lower()
    document = text.split()
    # print(document)
    # document = token_nize_for_tokenize(text)
    return  " ".join(document)
    # return " ".join([token.text for token in document])

def aspect_short_path(G, target):
    """"
    """
    d = nx.shortest_path_length(G, target=target)
    distance_list = []
    for node in G.nodes():
        try:
            distance_list.append(d[node])
        except KeyError:
            distance_list.append(-1)
    return distance_list

def dependency_adj_matrix(text):
    # text = "Great food but the service was dreadful !"
    seq_len = len((text).split())
    document = token_nize_for_tokenize(text)

    Syntactic_dependence = []
    # 创建三元组(piece_token,old_index,new_index)
    three_list = []
    distance_list = []

    matrix_dir = np.zeros([seq_len,seq_len]).astype('float32')

    matrix_undir = np.zeros([seq_len, seq_len]).astype('float32')

    for token in document:
        if token.i<seq_len:
            matrix_dir[token.i][token.i] = 1
            matrix_undir[token.i][token.i] = 1
            for child in token.children:
                if child.i < seq_len:
                    matrix_undir[token.i][child.i] = 1
                    matrix_undir[child.i][token.i] = 1
                    matrix_dir[child.i][token.i] = 1
                Syntactic_dependence.append([token.i, child.dep_.lower(), child.i])
            Syntactic_dependence.append([token.i, "selfcycle", token.i ])
    G = nx.from_numpy_matrix(matrix_undir)
    for i in range(seq_len):
        i_distance = np.array(aspect_short_path(G, i)).tolist()
        for index, distance in enumerate(i_distance):
            if distance == -1:
                i_distance[index] = 1000
        distance_list.append(i_distance)

    # displacy.serve(document, style='dep')

    return matrix_dir,matrix_undir,Syntactic_dependence,distance_list


def process(filename_path):

    lines = json.load(open(filename_path))
    idx2graph_dir = {}
    idx2graph_redir ={}
    idx2graph_undir ={}
    Syntactic_dependence_all = {}
    idx2positon = {}

    filename = filename_path
    # print(filename_list)
    # print(filename+"_"+'dir'+ '.graph')
    # exit()
    fout_dir= open(filename+'dir'+ '.graph', 'wb')
    print(fout_dir)
    # fout_redir = open(filename+"_"+depenpency_type+"redir"+ '.graph', 'wb')
    fout_undir = open(filename+'undir'+ '.graph', 'wb')
    dependency_analysis = open(filename+'.dependency','wb')
    fout_syntax_position = open(filename+'.syntaxPosition', 'wb')

    for i in range(0, len(lines)):
        # print(i)

        text = lines[i]['sentence']
        # print(text)
        input_text = tokenize(text)
        # print(input_text)
        # print(input_text)
        adj_matrix_dir,adj_matrix_undir,Syntactic_dependence,distance_list = dependency_adj_matrix(input_text)
        # print(adj_matrix_dir)
        # print(adj_matrix_undir)
        # print(Syntactic_dependence)
        # print(distance_list)
        # exit()
        idx2graph_dir[i] = adj_matrix_dir
        # idx2graph_redir[i] = adj_matrix_redir
        idx2graph_undir[i] = adj_matrix_undir

        Syntactic_dependence_all[i] = Syntactic_dependence
        #syntax_position_distance
        idx2positon[i] = distance_list
        print(i)

    pickle.dump(idx2graph_dir, fout_dir)
    pickle.dump(idx2graph_undir, fout_undir)
    # pickle.dump(idx2graph_redir,fout_redir)
    pickle.dump(Syntactic_dependence_all,dependency_analysis)
    pickle.dump(idx2positon,fout_syntax_position)
    # fout_dir.close()
    # fout_redir.close()
    fout_undir.close()
    dependency_analysis.close()
    fout_syntax_position.close()

if __name__ == '__main__':
    # process('14lap/train_triplets.pair')
    #
    # process('14lap/test_triplets.pair')
    # process('14lap/dev_triplets.pair')
    # exit()

    # print('################################')
    # process('./lap14/train.json')
    # process('./lap14/dev.json')
    # process('./lap14/test.json')
    # print('################################')
    # process('./res14/train.json')
    # process('./res14/dev.json')
    # process('./res14/test.json')
    # print('################################')
    # process('./res15/train.json')
    # process('./res15/dev.json')
    # process('./res15/test.json')
    print('################################')
    process('./res16/train.json')
    process('./res16/dev.json')
    process('./res16/test.json')

    # # fin= open("./datasets/acl-14-short-data/train.raw.dependency","rb")

