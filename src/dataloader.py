import os
import pdb
import pickle
import json
import random

import numpy as np
import torch
from scipy.sparse import csr_matrix

from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, rel_data, args, split):
        self.split = split
        self.args = args
        self.neg_num = self.args.neg_num
        self.ent_num = self.args.ent_num
        self.neg_num_list = []
        # pdb.set_trace()
        if args.ta:
            self.temp_neg, self.lf_neg, self.sf_neg = self.collect_neg(rel_data)
        self.latest_data, self.neg_mask = self.pre_process(rel_data[0], 'temp')
        self.lf_data, _ = self.pre_process(rel_data[1], 'lf')
        self.sf_data, _ = self.pre_process(rel_data[2], 'sf')


    def __getitem__(self, index):
        if self.split == 'test':
            latest_data = torch.from_numpy(self.latest_data[index].toarray()).float()
            lf_data = torch.from_numpy(self.lf_data[index].toarray()).float()
            sf_data = torch.from_numpy(self.sf_data[index].toarray()).float()
            neg_mask = torch.from_numpy(self.neg_mask[index]).float()
        else:
            latest_data = self.latest_data[index]
            lf_data = self.lf_data[index]
            sf_data = self.sf_data[index]
            neg_mask = self.neg_mask[index]

        return latest_data, lf_data, sf_data, neg_mask

    def __len__(self):
        return len(self.latest_data)

    def pre_process(self, data, mode):
        if self.split !='test':
            data = [data_item.toarray()[:self.neg_num+1] for data_item in data]
            if self.split == 'train':
                data = self.pad_train_negs(data, mode)
            data = np.stack(data, axis=0)
            data = torch.from_numpy(data).contiguous().float()
            neg_mask = data[:,1:,-1]
            self.neg_num_list += neg_mask.sum(1).tolist()
            data = data[:, :, :-1]
        else:
            data, neg_mask = self.pad_test_negs(data)
        return data, neg_mask

    def pad_train_negs(self, data_list, mode):
        output_list = []
        if self.args.ta:
            if mode == 'sf':
                pad_base = self.sf_neg
            elif mode == 'lf':
                pad_base = self.lf_neg
            elif mode == 'temp':
                pad_base = self.temp_neg
            pad_base_len = len(pad_base)

        for data in data_list:
            actual_neg = sum(data[1:, -1])
            pad_num = self.neg_num - actual_neg
            if pad_num > 0:
                data = data[:actual_neg+1]
                if self.args.ta:
                    # pad_array = np.repeat(data[0, :].reshape(1,-1), pad_num, axis=0)
                    # neg_time = np.arange(1, pad_num+1).reshape(-1, 1)
                    # pad_array = pad_array - neg_time
                    # pad_array[pad_array<0] = 0
                    # pad_label = np.sum(pad_array, axis=1)
                    # pad_label[pad_label>0] = 1
                    # pad_array[:,-1] = pad_label
                    # data = np.concatenate((data, pad_array), axis=0)

                    sample_id = random.sample(list(range(pad_base_len)), pad_num)
                    pad_array = pad_base[sample_id]
                    data = np.concatenate((data, pad_array), axis=0)

                else:
                    pad_array = np.zeros((pad_num, data.shape[1]))
                    data = np.concatenate((data, pad_array), axis=0)

            output_list.append(data)
        return output_list

    def collect_neg(self, rel_data):
        temp_data = np.concatenate([data_item.toarray()[1:] for data_item in rel_data[0]], axis=0)
        temp_data = temp_data[temp_data[:,-1] == 1]
        lf_data = np.concatenate([data_item.toarray()[1:] for data_item in rel_data[1]], axis=0)
        lf_data = lf_data[lf_data[:, -1] == 1]
        sf_data = np.concatenate([data_item.toarray()[1:] for data_item in rel_data[2]], axis=0)
        sf_data = sf_data[sf_data[:, -1] == 1]

        return temp_data, lf_data, sf_data



    def pad_test_negs(self, items):
        output_list = []
        neg_masks = []
        rule_dim = items[0].shape[1]
        neg_nums = [item.shape[0] for item in items]
        max_neg = self.args.ent_num
        for i, item in enumerate(items):
            neg_mask = np.ones(max_neg-1)
            item = item.toarray()
            neg_item = item[1:]
            neg_mask[:neg_item.shape[0]] = neg_item[:,-1]

            pad_num = max_neg - neg_nums[i]
            if pad_num > 0:
                pad_array = np.zeros((pad_num, rule_dim))
                pad_array = np.concatenate((item, pad_array), axis =0)

                neg_mask[-pad_num:] = 0
            else:
                pad_array = item
            pad_array = pad_array[:, :-1]
            row, col = np.nonzero(pad_array)
            values = pad_array[row, col]
            pad_csr = csr_matrix((values, (row, col)), shape=pad_array.shape)
            output_list.append(pad_csr)
            neg_masks.append(neg_mask)

        return output_list, neg_masks


def load_data(args, rel, split='train'):
    latest_path = '../output/{0}/{1}/{1}_{2}_latest.pkl'.format(args.dataset, split, rel)
    # latest_path = '../output/{0}/{1}1/LR_{1}_{2}.pkl'.format(args.dataset, split, rel)

    if os.path.isfile(latest_path):
        with open(latest_path, 'rb') as f:
            latest_data = pickle.load(f)
        lf_path = '../output/{0}/{1}/{1}_{2}_lf.pkl'.format(args.dataset, split, rel)
        with open(lf_path, 'rb') as f:
            lf_data = pickle.load(f)
        sf_path = '../output/{0}/{1}/{1}_{2}_sf.pkl'.format(args.dataset, split, rel)
        with open(sf_path, 'rb') as f:
            sf_data = pickle.load(f)
        return latest_data, lf_data, sf_data
    else:
        return None
