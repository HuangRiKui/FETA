# this file has been modified by authors of evaluation paper for TGL Neurips workshop
# all changes can be seen via git diff, and are marked with a comment containing "eval_paper_authors"

import time
import argparse
import pdb
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default="3", type=int)
parser.add_argument("--num_walks", "-n", default="100", type=int)
parser.add_argument("--transition_distr", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=1, type=int)
parser.add_argument("--seed", "-s", default=None, type=int)
parser.add_argument("--runnr", default=0, type=int) # modified eval_paper_authors for logging
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
rule_lengths = [i for i in range(1, rule_lengths+1)]
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]
exp_nr = parsed["runnr"] #eval_paper_authors for logging

dataset_dir = "../data/" + dataset + "/"

data = Grapher(dataset_dir)
temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
rl = Rule_Learner(temporal_walk.edges, data.id2relation, data.inv_relation_id, dataset)
all_relations = sorted(temporal_walk.edges)  # Learn for all relations


def learn_rules(i, num_relations, seed): # eval_paper_authors added seed
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        num_relations (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    if seed !=0: # modified eval_paper_authors 
        np.random.seed(seed) # modified eval_paper_authors 
    else: # modified eval_paper_authors 
        seed = "None" # modified eval_paper_authors 
        print("seed" , seed) # modified eval_paper_authors 

    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for length in rule_lengths:
            it_start = time.time()
            for _ in range(num_walks):
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_successful:
                    rl.create_rule(walk)
            # pdb.set_trace()
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            num_new_rules = num_rules[-1] - num_rules[-2]
            print(
                "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
                    i,
                    k - relations_idx[0] + 1,
                    len(relations_idx),
                    length,
                    it_time,
                    num_new_rules,
                )
            )

    return rl.rules_dict


start = time.time()
num_relations = len(all_relations) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations, seed) for i in range(num_processes) # modified eval_paper_authors pass seed 
)
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

rl.rules_dict = all_rules
rl.sort_rules_dict()
rl.save_rules()
rl.save_rules_verbalized()
rules_statistics(rl.rules_dict)