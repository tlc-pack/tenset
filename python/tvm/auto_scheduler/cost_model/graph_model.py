
""" Cost model based on graphNN """

import numpy as np
import time
import logging

from ..graph_feature import get_graph_from_measure_pairs, get_graph_from_states, get_graph_from_file
from .cost_model import PythonBasedModel

import dgl
import collections
import torch
import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.function as fn
import math
import copy
import matplotlib.pyplot as plt
import torch as th
from itertools import chain

logger = logging.getLogger('auto_scheduler')

Edge = collections.namedtuple("Edge", ["src", "dst", "feature"])
Node = collections.namedtuple("Node", ["node_type", "id", "feature"])


def compute_rmse(preds, labels):
    """Compute RMSE (Rooted mean square error)"""
    return np.sqrt(np.mean(np.square(preds - labels)))

class graphNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(graphNN, self).__init__()
        self.msg = torch.nn.Linear(node_dim + edge_dim, hidden_dim)
        self.conv1 = dglnn.TAGConv(hidden_dim + node_dim, hidden_dim)
        self.conv2 = dglnn.TAGConv(hidden_dim, hidden_dim)

        self.predict = torch.nn.Linear(hidden_dim, 1)

    def message_func(self, edges):
        return {'mid': F.relu(self.msg(torch.cat([edges.src['fea'], edges.data['fea']], 1)))}

    def forward(self, g):
        with g.local_scope():
            g.update_all(self.message_func, fn.sum('mid', 'h_neigh'))
            h = F.relu(self.conv1(g, torch.cat([g.ndata['fea'], g.ndata['h_neigh']], 1)))
            h = F.relu(self.conv2(g, h))
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')
            hg = self.predict(hg)
            return hg


class GraphModel(PythonBasedModel):

    def __init__(self):
        self.params = {
            'batch_size': 32,
            'itr_num': 100,
            'lr':  0.01,
            'hidden_dim': 20,
            'node_fea': 84,
            'edge_fea': 4,
        }

        self.graphNN = None

        super().__init__()

        # measurement input/result pairs
        self.inputs = []
        self.results = []
        self.few_shot_learning="base_only"

    def register_new_task(self, task):
        pass
        #workload_key = str(task.workload_key)
        #self.workload_embed_dict[workload_key] = get_workload_embedding(workload_key)

    def fit_base(self, train_set, valid_set=None, valid_train_set=None):
        if self.few_shot_learning == "local_only":
            self.base_model = None
        else:
            self.base_model = self._fit_a_model(train_set, valid_set, valid_train_set)

    def _fit_a_model(self, train_set, valid_set=None, valid_train_set=None):
        print("Fit a GNN. Train size: %d" % len(train_set))

        def build_graph(pair):
            #print(len(pair))
            #print(len(pair[0]))
            (src_cur, dst_cur, edge_fea), node_fea, normalized_throughput = pair
            g = dgl.graph((th.tensor(src_cur), th.tensor(dst_cur)))
            g.edata['fea'] = th.tensor(edge_fea).float()
            g.ndata['fea'] = node_fea
            return g, normalized_throughput

        pairs = list(train_set.features.values())

        # extract feature
        idx = np.random.permutation(len(pairs))
        train_pairs = list(chain(*[[build_graph(x) for x in pairs[i]] for i in idx]))
        print(train_pairs[0])
        train_batched_graphs, train_batched_labels = create_batch(train_pairs, self.params['batch_size'])

        self.graphNN = graphNN(self.params['node_fea'], self.params['edge_fea'], self.params['hidden_dim']).float()
        opt = torch.optim.Adam(self.graphNN.parameters(), lr=self.params['lr'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
        n = len(train_batched_graphs)
        loss_func = torch.nn.MSELoss()

        print('Learning rate: {} batch size: {}'.format(self.params['lr'], self.params['batch_size']))

        for epoch in range(self.params['itr_num']):
            total_loss = 0
            preds = []
            labels = []
            tic = time.time()
            for i in range(n):
                opt.zero_grad()
                prediction = self.graphNN(train_batched_graphs[i])
                loss = loss_func(prediction, train_batched_labels[i].unsqueeze(1))
                total_loss += loss.detach().item() * self.params['batch_size']
                loss.backward()
                opt.step()
                preds = preds + prediction.squeeze().tolist()
                labels = labels + train_batched_labels[i].squeeze().tolist()
            epoch_loss = compute_rmse(np.array(preds), np.array(labels))
            print("Time spent in last epoch: %.2f" % (time.time() - tic))
            if epoch % 100 == 0:
                scheduler.step()
            if epoch % 10 == 0:
                print('Epoch {} | loss {:.4f}'.format(epoch, epoch_loss))
        
        return self.graphNN

    def predict(self, dataset):
        if self.few_shot_learning in ["base_only", "fine_tune_mix_task", "fine_tune_per_task", "MAML"]:
            return self._predict_a_dataset(self.base_model, dataset)
        elif self.few_shot_learning in ["local_only_mix_task", "local_only_per_task"]:
            ret = {}
            for task in dataset.tasks():
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = local_preds
            return ret
        elif self.few_shot_learning in ["plus_mix_task", "plus_per_task"]:
            base_preds = self._predict_a_dataset(self.base_model, dataset)
            ret = {}
            for task in dataset.tasks():
                if task not in self.local_model and self.few_shot_learning == "plus_mix_task":
                    self.local_model[task] = list(self.local_model.values())[0]
                local_preds = self._predict_a_task(self.local_model[task], task, dataset.features[task])
                ret[task] = base_preds[task] + local_preds
            return ret
        else:
            raise ValueError("Invalid few show learing: " + self.few_shot_learning)
    
    def _predict_a_dataset(self, model, dataset):
        ret = {}
        for task, features in dataset.features.items():
            ret[task] = self._predict_a_task(model, task, features)
        return ret

    def _predict_a_task(self, model, task, features):
        def build_graph(pair):
            #print(len(pair))
            #print(len(pair[0]))
            if len(pair) == 3: (src_cur, dst_cur, edge_fea), node_fea, _ = pair
            else: (src_cur, dst_cur, edge_fea), node_fea = pair
            g = dgl.graph((th.tensor(src_cur), th.tensor(dst_cur)))
            g.edata['fea'] = th.tensor(edge_fea).float()
            g.ndata['fea'] = node_fea
            return g

        graphs = [build_graph(x) for x in features]
        tic = time.time()
        batched_graphs = dgl.batch(graphs)
        preds = model(batched_graphs).squeeze().tolist()
        print("prediction time: %.2f" % (time.time() - tic))
        return preds

    def save(self, file_name: str):
        print("saving to: "+ file_name)
        torch.save(self.graphNN.state_dict(), file_name)

    def load(self, file_name: str):
        print("loading from: "+file_name)
        self.graphNN = graphNN(self.params['node_fea'], self.params['edge_fea'], self.params['hidden_dim']).float()
        self.graphNN.load_state_dict(torch.load(file_name))


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def create_batch(graph_pairs, batch_size):
    batched_graphs = []
    batched_labels = []
    for i in range(0, len(graph_pairs), batch_size):
        if i+batch_size >= len(graph_pairs):
            g, l = collate(graph_pairs[i:])
            batched_graphs.append(g)
            batched_labels.append(l)
        else:
            g, l = collate(graph_pairs[i:i+batch_size])
            batched_graphs.append(g)
            batched_labels.append(l)
    return batched_graphs, batched_labels
