# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Train a compute graph embedding model """
import numpy as np
from math import sqrt
import argparse
import os
import glob
import pickle
import json

import stellargraph as sg
import tensorflow as tf
from tensorflow import keras
import networkx as nx
from scipy.sparse import coo_matrix
from karateclub import LDP
from common import load_and_register_tasks

from tvm.auto_scheduler.workload_registry import workload_key_to_tensors, workload_key_to_dag
from tvm.auto_scheduler.measure_record import RecordReader


SHAPE_LENGTH = 6

from tensorflow.python.keras import backend as K

# adjust values to your needs
#config = tf.compat.v1.ConfigProto(device_count = {'GPU': 4})
#sess = tf.compat.v1.Session(config=config) 
#K.set_session(sess)

def node_match(node1, node2):
    feat1 = node1["feature"]
    feat2 = node2["feature"]
    return list(feat1) == list(feat2)

def graph_distance(graph1, graph2):
    return nx.graph_edit_distance(graph1.to_networkx(feature_attr="feature"),graph2.to_networkx(feature_attr="feature"), node_match=node_match)
    #spec1 = nx.laplacian_spectrum(graph1.to_networkx(feature_attr=None))
    #spec2 = nx.laplacian_spectrum(graph2.to_networkx(feature_attr=None))
    #k = min(len(spec1), len(spec2))
    #return np.linalg.norm(spec1[:k] - spec2[:k])


def get_prime_factors(n):
    primfac = {}
    if n == 1:
        primfac[1] = 1
        return primfac
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            if d in primfac:
                primfac[d] += 1
            else:
                primfac[d] = 1
            n //= d
        d += 1
    if n > 1:
        primfac[n] = 1
    return primfac

PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

class GraphEmbeddingModel:
    def __init__(self, model):
        self.task_embeddings = {} # Dict[workload_key -> embedding vector]
        # self.tensor_type_one_hot_dict = {} # Dict[tensor name -> id]
        self.tensor_type_one_hot_dict = {"virtual node": 0} # Dict[tensor name -> id]
        self.tensor_type_idx = 0
        self.model_type = model

    def create_model(self, graphs):
        if self.model_type == 'LDP':
            self.model = LDP()
        elif self.model_type == 'UGraphEmb':
            self.generator = sg.mapper.PaddedGraphGenerator(graphs)
            gc_model = sg.layer.GCNSupervisedGraphClassification(
                [64, 32], ["relu", "relu"], self.generator, pool_all_layers=True
            )
            inp1, out1 = gc_model.in_out_tensors()
            inp2, out2 = gc_model.in_out_tensors()
            vec_distance = tf.norm(out1 - out2, axis=1)
            self.pair_model = keras.Model(inp1 + inp2, vec_distance)
            self.model = keras.Model(inp1, out1)
        else:
            self.model = None


    def fit_UGraphEmb(self, graphs):
        graphs = graphs[:100]
        graph_idx = np.random.RandomState(0).randint(len(graphs), size=(100, 2))
        targets = [graph_distance(graphs[left], graphs[right]) for left, right in graph_idx]
        train_gen = self.generator.flow(graph_idx, batch_size=10, targets=targets)
        self.pair_model.compile(keras.optimizers.Adam(1e-2), loss="mse")
        history = self.pair_model.fit(train_gen, epochs=300, verbose=0)
        sg.utils.plot_history(history)

    def fit(self, graphs):
        self.model.fit(graphs)

    def get_embedding(self, workload_keys):
        embedding = self.model.get_embedding()
        for i in range(len(workload_keys)):
            self.task_embeddings[workload_keys[i]] = embedding[i]
        print(len(embedding[0]))

    def get_UGraphEmb(self, workload_keys, graphs):
        embedding = self.model.predict(self.generator.flow(graphs))
        for i in range(len(workload_keys)):
            self.task_embeddings[workload_keys[i]] = embedding[i]
        print(len(embedding[0]))

    def traverse_and_build_the_graph(self, tensors):
        visited = set()
        G = nx.Graph()

        node_dict = {}
        node_idx = 0

        def build_node_dict(t):
            if t.ndim == 0:
                return
            if t in node_dict:
                return
            nonlocal node_idx
            node_dict[t] = node_idx
            feat = [s.value for s in t.shape]
            feat = np.array(feat, dtype=np.float32)
            feat = np.pad(feat, (0, SHAPE_LENGTH - len(feat)), 'constant', constant_values=0.0)

            feat_verbose = []
            for sh in feat:
                factors = get_prime_factors(sh)
                prime_one_hot = np.zeros(13)
                for key in factors:
                    if key > 31:
                        idx = 12
                    else:
                        idx = PRIMES.index(key)
                    prime_one_hot[idx] = factors[key]
                feat_verbose += list(prime_one_hot)

            if t.name not in self.tensor_type_one_hot_dict:
                self.tensor_type_one_hot_dict[t.name] = self.tensor_type_idx
                self.tensor_type_idx += 1
            G.add_node(node_idx, feature=feat_verbose, name=t.name)
            node_idx += 1
            for x in t.op.input_tensors:
                build_node_dict(x)

        def add_edges(t):
            if t in visited:
                return
            for x in t.op.input_tensors:
                if x.ndim != 0:
                    G.add_edge(node_dict[x], node_dict[t])
                add_edges(x)
            visited.add(t)

        for t in tensors:
            print(t)
            build_node_dict(t)

        for t in tensors:
            add_edges(t)

        return G

    def add_tensor_type_one_hot(self, graphs):
        for graph in graphs:
            for node, feat in graph.nodes(data=True):
                tensor_type_id = self.tensor_type_one_hot_dict.get(feat['name'])
                type_embedding = np.zeros(len(self.tensor_type_one_hot_dict), dtype=np.float32)
                type_embedding[tensor_type_id] = 1.0
                feat['feature'] = list(feat['feature']) + list(type_embedding)
        return graphs

    def load(self, embedding_file):
        self.task_embeddings = pickle.load(open(embedding_file, 'rb'))

    def save(self, embedding_file):
        pickle.dump(self.task_embeddings, open(embedding_file, 'wb'))


if __name__ == "__main__":
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-file", type=str, default='task_embeddings.pkl')
    parser.add_argument("--use-saved-graphs", action='store_true')
    parser.add_argument("--embedding-type", type=str, default='LDP')
    parser.add_argument("--log-dir", type=str, default='dataset/measure_records/e5-2673')
    args = parser.parse_args()

    graph_embedding_model = GraphEmbeddingModel(args.embedding_type)
    i = 0
    if not args.use_saved_graphs:
        load_and_register_tasks()
        graphs = []
        workload_keys = []
        directory = args.log_dir

        for filename in tqdm(os.listdir(directory)):
            if filename.endswith(".json"):
                for inp, _ in RecordReader(f"{directory}/{filename}"):
                    workload_key = inp.task.workload_key
                    workload = json.loads(workload_key)
                    if workload[0] not in workload_keys:
                        graph = graph_embedding_model.traverse_and_build_the_graph(
                            workload_key_to_tensors(workload_key))
                        graphs.append(graph)
                        workload_keys.append(workload[0])

        graphs = graph_embedding_model.add_tensor_type_one_hot(graphs)
        pickle.dump((graphs, workload_keys), open("all_task_graphs.pkl", 'wb'))
    else:
        graphs, workload_keys = pickle.load(open("all_task_graphs.pkl", 'rb'))

    if graph_embedding_model.model_type == 'UGraphEmb':
        sg_graphs = [sg.StellarGraph.from_networkx(graph, node_features="feature") for graph in graphs]
        graph_embedding_model.create_model(sg_graphs)
        graph_embedding_model.fit_UGraphEmb(sg_graphs)
        graph_embedding_model.get_UGraphEmb(workload_keys, sg_graphs)
    else:
        graph_embedding_model.create_model(graphs)
        graph_embedding_model.fit(graphs)
        graph_embedding_model.get_embedding(workload_keys)

    graph_embedding_model.save('task_embeddings.pkl')
