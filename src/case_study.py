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
parser.add_argument("--dataset", "-d", default="ICEWS18", type=str)
parser.add_argument("--test_file", default="test", type=str)
parser.add_argument("--model", "-m", default="FETA", type=str)
parser.add_argument("--model_path", default="../output/ICEWS18/results/gamma_0.7_FETA/models/", type=str)
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

# with open('../output/ICEWS18/14_2_18_rel_dict.json') as f:
#     rel14_2_rel18 = json.load(f)
#     rel14_2_rel18 = { int(k):int(v) for k,v in rel14_2_rel18.items()}
#     convert_dict = {v:k for k,v in rel14_2_rel18.items()}

dataset_dir = "../data/" + args.dataset + "/"
model_dir = args.model_path
quadruple_data = Grapher(dataset_dir)
rels = list(quadruple_data.id2relation.keys())
rels.sort()
convert_dict = {rel:rel for rel in rels}
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
                # pos_scores = torch.sum(pos_scores, dim=-1)
                # neg_scores = torch.sum(neg_scores, dim=-1)
            else:
                pos_scores, neg_scores = model(latest_data)
            # pos_masks = torch.sum(latest_data[:,0,:], dim=-1).unsqueeze(-1)
            # pos_masks[pos_masks!=0] = 1
            # pos_scores = pos_scores * pos_masks
            pos_scores = pos_scores.reshape(neg_scores.shape[0],-1)
            neg_scores = neg_scores * neg_masks
            # put pos behind neg to avoid unreasonable rank under same score
            scores = torch.cat([neg_scores, pos_scores], dim=1)
            sort_scores = torch.argsort(scores, dim=1, descending=True)
            ranks = (sort_scores == sort_scores.shape[1] - 1).nonzero()[:, 1] + 1
            # ranks1 = (sort_scores == sort_scores.shape[1] - 1).nonzero()[:, 1] + 1
            # scores = torch.cat([pos_scores, neg_scores], dim=1)
            # sort_scores = torch.argsort(scores, dim=1, descending=True)
            # ranks = (sort_scores == 0).nonzero()[:, 1] + 1
            # ranks =(ranks1+ranks2)/2.0
            # pdb.set_trace()

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
rel =17

model_path = model_dir + 'rel_{}.pth'.format(rel)
rel_model_dict = torch.load(model_path)
pdb.set_trace()

