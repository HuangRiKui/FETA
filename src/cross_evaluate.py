import logging
import os
import pdb
import pickle
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader

from grapher import Grapher
from dataloader import Dataset
from models import FETA
from temporal_walk import store_edges
import rule_application as ra

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
parser.add_argument("--test_file", default="test", type=str)
parser.add_argument("--model", "-m", default="FETA", type=str)
parser.add_argument("--model_path", default="../output/ICEWS14/results/results/gamma_0.8_FETA/models/", type=str)
parser.add_argument("--beta", "-b", default=0.1, type=float)
parser.add_argument("--gamma", "-g", default=0.5, type=float)
parser.add_argument("--neg_num", "-n", default=100, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--valid_method", default="mrr", type=str)
parser.add_argument("--ta", action = "store_true")
parser.add_argument("--cuda", action = "store_true")
parser.add_argument("--save_name", default="", type=str)
args = parser.parse_args()


# with open('../output/ICEWS14/0515_2_14_rel_dict.json') as f:
#     rel0515_2_rel14 = json.load(f)
#     rel0515_2_rel14 = { int(k):int(v) for k,v in rel0515_2_rel14.items()}
#     convert_dict = {v:k for k,v in rel0515_2_rel14.items()}

with open('../output/ICEWS18/14_2_18_rel_dict.json') as f:
    rel14_2_rel18 = json.load(f)
    rel14_2_rel18 = { int(k):int(v) for k,v in rel14_2_rel18.items()}
    convert_dict = {v:k for k,v in rel14_2_rel18.items()}

dataset_dir = "../data/" + args.dataset + "/"
model_dir = args.model_path
quadruple_data = Grapher(dataset_dir)
rels = list(quadruple_data.id2relation.keys())
rels.sort()
#convert_dict = {rel:rel for rel in rels}
args.ent_num = len(quadruple_data.entity2id)
all_test_num = len(quadruple_data.test_idx)
learn_edges = store_edges(quadruple_data.train_idx)

def evaluate(args, test_loader, model, mode='mrr'):
    model = model.eval()
    valid_loss = 0
    epoch_num = 0
    rank_list = []
    with torch.no_grad():
        for i_batch, (latest_data, lf_data, sf_data, neg_masks) in enumerate(test_loader):
            if args.cuda:
                latest_data = latest_data.cuda()
                lf_data = lf_data.cuda()
                sf_data = sf_data.cuda()
                neg_masks = neg_masks.cuda()
            if args.model == 'FETA':
                pos_scores, neg_scores = model(latest_data, lf_data, sf_data)
            else:
                pos_scores, neg_scores = model(latest_data)
            pos_scores = pos_scores.reshape(neg_scores.shape[0],-1)
            neg_scores = neg_scores * neg_masks
            # put pos behind neg to avoid unreasonable rank under same score
            scores = torch.cat([neg_scores, pos_scores], dim=1)
            sort_scores = torch.argsort(scores, dim=1, descending=True)
            ranks = (sort_scores == sort_scores.shape[1] - 1).nonzero()[:, 1] + 1

            rank_list.append(ranks)

    if epoch_num>0:
        valid_loss /= epoch_num

    if mode=='mrr':
        rank_list = torch.cat(rank_list, dim=0)

    return rank_list, valid_loss

set_name = args.save_name + '_' + args.model
log_dir = '../output/{0}/results/{1}/'.format(args.dataset, set_name)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = log_dir + '{}.log'.format(set_name)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=log_file,
    filemode='w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

eval_array = np.zeros(4)
cover_count = 0
all_count = 0

args.ent_num = len(quadruple_data.entity2id)
all_test_num = len(quadruple_data.test_idx)
rules_dict = json.load(open("../output/"+args.dataset+"/rules_dict.json"))
rules_dict = {int(k): v for k, v in rules_dict.items()}
rule_lengths = [1,2,3]
rules_dict = ra.filter_rules(rules_dict, min_conf=0.01, min_body_supp=2, rule_lengths=rule_lengths)

# model_path = model_dir + 'rel_{}.pth'.format(4)
# rel_model_dict = torch.load(model_path)
# pdb.set_trace()
# model.load_state_dict(rel_model_dict)

# pdb.set_trace()

for source_rel in rels:

    test_path = '../output/{0}/test_cross/test_{1}_latest.pkl'.format(args.dataset, source_rel)
    if os.path.isfile(test_path):
        with open(test_path, 'rb') as f:
            latest_data = pickle.load(f)
        test_path = '../output/{0}/test_cross/test_{1}_lf.pkl'.format(args.dataset, source_rel)
        with open(test_path, 'rb') as f:
            lf_data = pickle.load(f)
        test_path = '../output/{0}/test_cross/test_{1}_sf.pkl'.format(args.dataset, source_rel)
        with open(test_path, 'rb') as f:
            sf_data = pickle.load(f)
        test_data=(latest_data, lf_data, sf_data)
    else:
        test_data = None

    test_data_num = quadruple_data.test_idx[quadruple_data.test_idx[:, 1] == source_rel].shape[0]
    all_count += test_data_num
    if test_data!= None and len(test_data) >0 and source_rel in convert_dict.keys():
        test_num = len(test_data[0])
        cover_count += test_num
        rule_dim = test_data[0][0].shape[1]

        rel = convert_dict[source_rel]
        model_path = model_dir + 'rel_{}.pth'.format(rel)
        if os.path.exists(model_path):
            test_dataset = Dataset(test_data, args, split='test')
            test_loader = DataLoader(test_dataset, batch_size=16)
            model = FETA(rule_dim, args)
            rel_model_dict = torch.load(model_path)
            model.load_state_dict(rel_model_dict)
            if args.cuda:
                model = model.cuda()

            ranks, valid_loss = evaluate(args, test_loader, model, mode='mrr')
            mrr = torch.sum(1.0 / ranks).item()
            h1 = torch.sum(ranks <= 1).item()
            h3 = torch.sum(ranks <= 3).item()
            h10 = torch.sum(ranks <= 10).item()
            eval_array += np.array([mrr, h1, h3, h10])
            logging.info('rel_id: {}, MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, test_num: {}'.format(
                rel, mrr / test_num, h1 / test_num, h3 / test_num, h10 / test_num, test_num))
        else:
            logging.info('rel_id: {}, MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, test_num: {}'.format(
                rel, 0, 0, 0, 0, 0))
        logging.info(
            'Accum: MRR: {:.4f}, H@1: {:.4f}, H@3: {:.4f}, H@10: {:.4f}, count:{}/{}'.format(eval_array[0] / all_count,
                                                                                             eval_array[1] / all_count,
                                                                                             eval_array[2] / all_count,
                                                                                             eval_array[3] / all_count,
                                                                                             all_count, all_test_num))
        logging.info('-------------------------------------------------------------------------------------')
        logging.info('                                                                                     ')

logging.info('MRR:  {:.5f}'.format(eval_array[0] / all_test_num))
logging.info('H@1:  {:.5f}'.format(eval_array[1] / all_test_num))
logging.info('H@3:  {:.5f}'.format(eval_array[2] / all_test_num))
logging.info('H@10: {:.5f}'.format(eval_array[3] / all_test_num))

