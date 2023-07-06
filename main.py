#coding utf-8

import json, os
import random
import argparse
import codecs
import torch
import torch.nn.functional as F
from tqdm import trange
import time

from data_old import load_data_instances, DataIterator
from model import Syntax_Transformer_BERTModel
import utils_old
from syntactic_utils import build_dependency_matrix,build_position_matrix,build_positionizer,build_dependencyizer
from models.Star_Dpendency_Trans import BertForSequenceClassification
import  numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args,position_tokenizer,dependency_tokenizer,dependency_embedding,position_embedding):

    # load dataset
    train_path = args.prefix + args.dataset + '/train.json'
    dev_path = args.prefix + args.dataset + '/dev.json'
    test_path = args.prefix + args.dataset + '/test.json'

    instances_train = load_data_instances(train_path, position_tokenizer, dependency_tokenizer, args)
    instances_dev = load_data_instances(dev_path, position_tokenizer, dependency_tokenizer, args)
    instances_test = load_data_instances(test_path,  position_tokenizer, dependency_tokenizer, args)


    devset = DataIterator(instances_dev, args)
    testset = DataIterator(instances_test, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    args.dependency_embedding = dependency_embedding
    args.position_embedding = position_embedding
    model =BertForSequenceClassification.from_pretrained(args.pretrained_weights,  num_labels=args.class_num, cus_config=args).to(args.device)
    # model =Syntax_Transformer_BERTModel(dependency_embedding,position_embedding,args).to(args.device)
    # print(model)
    # parameters = list(model.parameters())
    # parameters = filter(lambda x: x.requires_grad, parameters)

    optimizer = torch.optim.Adam([{"params":model.parameters(),"lr":5e-5},
                                 ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)

    best_joint_f1 = 0
    best_joint_precision=0
    best_joint_recall=0
    best_joint_epoch = 0
    test_f1 = 0
    test_p = 0
    test_r = 0
    best_test_model = None
    f_out = codecs.open('log/'  + args.dataset + '_val.txt', 'a+', encoding="utf-8")
    model_path = args.model_dir + args.model + args.task + args.dataset + args.tag_type + '.pt'

    for i in range(args.epochs):
        random.shuffle(instances_train)
        random.shuffle(instances_train)
        trainset = DataIterator(instances_train, args)
        print('Epoch:{}'.format(i))
        train_all_loss = 0.0
        all_token_length =0
        dependency_length = 0
        number = 0
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, token_ranges,\
               syntactic_position_datas,adj_index_undir_all,adj_dependency_index_all,adj_dependency_mask_all, \
            batch_dependency_length,aspect_tags, opinion_tags, tags = trainset.get_batch(j)

        #     model_out = model(tokens,lengths,syntactic_position_datas,adj_index_undir_all,adj_dependency_index_all,adj_dependency_mask_all,batch_dependency_length, masks)
        #     # exit()
            number += 1
        #
            all_token_length +=tokens.shape[-1]
            dependency_length+=torch.max(batch_dependency_length).cpu().numpy()
            # print(dependency_length)
            # print(all_token_length)
        #     exit()
        #
        #     # exit()
        #     preds = model_out.logits
        #     preds_flatten = preds.reshape([-1, preds.shape[3]])
        #     tags_flatten = tags.reshape([-1])
        #
        #     loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
        #     train_all_loss += loss
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        # scheduler.step()
        print("average length of token:{0}".format(all_token_length/number))
        print("average length of dependency:{0}".format(dependency_length/number))
        exit()
        print('this epoch train loss :{0}'.format(train_all_loss))
        # print("------------------this is train result-------------------------------------")
        # _, _, _, _ = eval(model, trainset, args)

        print("------------------this is dev result-------------------------------------")
        joint_precision, joint_recall, joint_f1,dev_loss = eval(model, devset, args)
        print("------------------this is test result-------------------------------------")
        test_joint_precision, test_joint_recall, test_joint_f1, _ = eval(model, testset, args)
        if test_joint_f1 > test_f1:
            test_f1 = test_joint_f1
            test_p = test_joint_precision
            test_r = test_joint_recall
            best_test_model = model
            best_joint_epoch = i
        print("11111111111111")

        #测试dev
        if joint_f1 > best_joint_f1:
            best_joint_f1 = joint_f1
            torch.save(model, model_path)
        print('this poch:\t dev {} loss: {:.5f}\n\n'.format(args.task, dev_loss))

    best_test_model_path = args.model_dir + args.model + args.task + args.dataset+args.tag_type + "best_test_f1" + str(
        test_f1) + '.pt'
    torch.save(best_test_model, best_test_model_path)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    f_out.write('time:{0}\n'.format(time_str))
    arguments = " "
    for arg in vars(args):
        if arg is "dependency_embedding":
            continue
        elif arg is "position_embedding":
            continue
        else:
            arguments += '{0}: {1} '.format(arg, getattr(args, arg))


    # yong dev_test来测试
    test_model = torch.load(model_path).to(args.device)
    dev_for_test_precision, dev_for_test_recall,dev_for_test_f1, _ = eval(test_model, testset, args)
    best_dev_for_test_model_path = args.model_dir + args.model + args.task + args.dataset + args.tag_type + "best_dev_for_test_f1" + str(
        dev_for_test_f1) + '.pt'
    torch.save(test_model, best_dev_for_test_model_path)

    f_out.write(arguments)
    f_out.write("\n")
    f_out.write('dev_max_test_acc: {0}, dev_max_test_recall:{1}, dev_max_f1: {2}\n'.format(dev_for_test_precision , dev_for_test_recall, dev_for_test_f1))

    f_out.write('best_joint_epoch:{0}, test_max_test_acc: {1}, test_max_test_recall:{2}, test_max_test_f1: {3}\n'.format(best_joint_epoch,test_p,test_r, test_f1))
    f_out.write("\n")
    f_out.close()

    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))
    print('max test precision:{} ----- recall:{}-------- f1:{}'.format(str(test_p), str(test_r), str(test_f1)))



def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        dev_loss =0.0
        for i in range(dataset.batch_count):
            sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, \
            syntactic_position_datas, adj_index_undir_all, adj_dependency_index_all, adj_dependency_mask_all, \
            batch_dependency_length, aspect_tags, opinion_tags, tags,dependency_masks = dataset.get_batch(i)
            preds = model(sentence_ids, tokens, lengths, syntactic_position_datas, adj_index_undir_all,
                          adj_dependency_index_all, adj_dependency_mask_all, batch_dependency_length, masks)
            if sentence_ids[0] == "33069925#747092#3":
                for i in dependency_masks:
                    print(i)
                # exit()
            prediction = preds.logits
            prediction_argmax = torch.argmax(prediction, dim=3)
            tags_flatten = tags.reshape([-1])
            prediction_flatten = prediction.reshape([-1, prediction.shape[3]])
            dev_loss = dev_loss + F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)

            prediction_padded = torch.zeros(prediction.shape[0], args.max_sequence_len, args.max_sequence_len)
            prediction_padded[:, :prediction_argmax .shape[1], :prediction_argmax .shape[1]] =prediction_argmax

            tags_padded = torch.zeros((prediction.shape[0], args.max_sequence_len, args.max_sequence_len))
            tags_padded[:, :prediction_argmax.shape[1], :prediction_argmax.shape[1]] = tags

            all_preds.append(prediction_padded)
            all_labels.append(tags_padded)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)


        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils_old.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1, dev_loss


def test(args):
    print("Evaluation on testset:")
    # model_path = args.model_dir + args.model + args.task + args.dataset+args.tag_type + '.pt'
    model_path = "./savemodel/rest_lap14/Sy_BERTtripletres14GTSbest_dev_for_test_f10.7232049947970863.pt"
    model = torch.load(model_path).to(args.device)
    model.eval()
    test_path = args.prefix + args.dataset + '/test.json'
    instances = load_data_instances(test_path, position_tokenizer, dependency_tokenizer,args)
    testset = DataIterator(instances, args)

    eval(model, testset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="view_data", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=6,
                        help='label number')
    parser.add_argument('--dependency_embed_dim', type=int, default=100,
                        )
    parser.add_argument('--position_embed_dim', type=int, default=100,
                        )
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--alpha_adjacent', type=float, default=0.4)
    parser.add_argument('--nhops', type=int, default=2,
                        help='inference times')

    parser.add_argument('--weight_edge', type=float, default=0.7)

    parser.add_argument('--decline', type=int, default=70, help="number of epochs to decline")
    parser.add_argument('--model', type=str, default="Sy_BERT", help="number of epochs to decline")
    parser.add_argument('--tag_type', type=str, default="GTS")
    parser.add_argument('--SD_begin_layer', type=int, default=10)

    parser.add_argument('--pretrained_weights', default="bert-base-uncased", type=str)

    parser.add_argument('--K_alpha', default=0.3, type=float)
    parser.add_argument('--V_alpha', default=0.3, type=float)


    args = parser.parse_args()
    setup_seed(args.seed)

    position_tokenizer = build_positionizer(args.prefix + args.dataset)
    dependency_tokenizer = build_dependencyizer(args.prefix + args.dataset)
    dependency_embedding = build_dependency_matrix(dependency_tokenizer.dependency2idx,
                                                   args.dependency_embed_dim, args.prefix + args.dataset,
                                                   "dependency")
    # print(dependency_embedding.shape)
    # exit()
    position_embedding = build_position_matrix(position_tokenizer.position2idx,
                                               args.position_embed_dim, args.prefix + args.dataset,
                                               "position")

    if args.mode == 'train':
        # train(args,position_tokenizer,dependency_tokenizer,dependency_embedding,position_embedding)
        test(args)
    else:
         test(args)
