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

""""
Python API for Feature extraction. The extracted features vector are used by cost models.

We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
so we call this feature as "per-store" feature.
The cost model also does prediction for each BufferStoreNode statement and aggregates
the predicted score of each BufferStoreNode as the score of a TIR Stmt.

The feature specification is defined by `src/auto_scheduler/feature.cc::FeatureSet`
"""

from typing import List, Tuple, Union, Optional
import struct
import collections
import numpy as np
import torch as th

from .loop_state import State, StateObject
from .measure import MeasureInput, MeasureResult
from . import _ffi_api

# The maximum number of extracted buffers for one statement
DEFAULT_MAX_N_BUFS = 5

# The length of the feature vector
DEFAULT_FEATURE_VEC_LEN = 164

# The size of int and float in bytes
SIZE_OF_INT32 = 4
SIZE_OF_FLOAT32 = 4

Edge = collections.namedtuple("Edge", ["src", "dst", "feature"])
Node = collections.namedtuple("Node", ["node_type", "id", "feature"])

def deserialize_graph(byte_arr, no_label=False) -> Tuple[List[Edge], List[Node], List, List]:

    pairs = []
    graphs = []
    edge_feature_len = 4
    node_feature_len = 1324

    # unpack size vector
    offset = 0
    n = struct.unpack_from("1i", byte_arr, offset=offset)[0]
    offset += SIZE_OF_INT32

    sizes = struct.unpack_from("%di" % (n+n+3), byte_arr, offset=offset)
    offset += SIZE_OF_INT32 * (n+n+3)

    #print(sizes[:n])

    for size in sizes[:n]:
        src_cur = []
        dst_cur = []
        fea_cur = []
        # unpack edge list
        for i in range(size):
            src = struct.unpack_from("1i", byte_arr, offset=offset)[0]
            offset += SIZE_OF_INT32
            dst = struct.unpack_from("1i", byte_arr, offset=offset)[0]
            offset += SIZE_OF_INT32
            feature = struct.unpack_from("%df" % edge_feature_len, byte_arr, offset=offset)
            offset += SIZE_OF_FLOAT32 * edge_feature_len
            src_cur.append(src)
            dst_cur.append(dst)
            fea_cur.append(feature)

        #g = dgl.graph((th.tensor(src_cur), th.tensor(dst_cur)))
        #g.edata['fea'] = th.tensor(fea_cur).float()
        g = (src_cur, dst_cur, fea_cur)
        if no_label:
            graphs.append([g])
        else:
            pairs.append([g])

    idx = 0
    for size in sizes[n:-3]:
        # unpack node list
        fea_cur = []
        for i in range(size):
            offset += SIZE_OF_INT32*2
            feature = struct.unpack_from("%df" % node_feature_len, byte_arr, offset=offset)
            offset += SIZE_OF_FLOAT32 * node_feature_len
            fea_cur.append(feature)

        """
        if no_label:
            graphs[idx].ndata['fea'] = th.tensor(fea_cur).float()
        else:
            pairs[idx][0].ndata['fea'] = th.tensor(fea_cur).float()
        """
        if no_label:
            graphs[idx].append(th.tensor(fea_cur).float())
        else:
            pairs[idx].append(th.tensor(fea_cur).float())
        idx += 1

    # unpack normalized_throughputs
    m = sizes[-3]
    normalized_throughputs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    for i in range(len(normalized_throughputs)):
        pairs[i].append(th.tensor(normalized_throughputs[i], dtype=th.float32))
    offset += m * SIZE_OF_INT32

    # unpack task_ids
    m = sizes[-2]
    task_ids = struct.unpack_from("%di" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_INT32

    # unpack min_costs
    m = sizes[-1]
    min_costs = struct.unpack_from("%df" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_FLOAT32

    #print('unpack successful')
    if no_label:
        return graphs, normalized_throughputs, np.array(task_ids), min_costs
    else:
        return pairs, normalized_throughputs, np.array(task_ids), min_costs

def get_per_store_feature_names(max_n_bufs: Optional[int] = None) -> List[str]:
    """Get the name of every element in the feature vector. Use this for debug and inspection.

    Parameters
    ----------
    max_n_bufs: int
        The maximum number of extracted buffers for one statement

    Returns
    -------
    names: List[str]
        The names of elements in the flatten feature vector
    """
    return _ffi_api.GetPerStoreFeatureNames(max_n_bufs or DEFAULT_MAX_N_BUFS)

def get_graph_from_states(states: List[Union[State, StateObject]],
                          task: "SearchTask",
                          max_n_bufs: int = None,
                          no_label = False) -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(states[0], State):
        state_objects = [s.state_object for s in states]
    elif isinstance(states[0], StateObject):
        state_objects = states
    byte_arr = _ffi_api.GetGraphFromStates(
        state_objects, task, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return deserialize_graph(byte_arr, no_label)

def get_graph_from_file(filename: str,
                        n_lines: int,
                        max_n_bufs: int = None) \
        -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """Get per_stmt features from a log file"""

    byte_arr = _ffi_api.GetGraphFromFile(
        filename, n_lines, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return deserialize_graph(byte_arr)

def get_graph_from_measure_pairs(inputs: List[MeasureInput],
                                 results: List[MeasureResult],
                                 skip_first_n_feature_extraction: int = 0,
                                 max_n_bufs: int = None) \
        -> Tuple[List, np.ndarray, np.ndarray, np.ndarray]:
    """Get per_stmt features from measurement pairs"""
    byte_arr = _ffi_api.GetGraphFromMeasurePairs(
        inputs, results, skip_first_n_feature_extraction, max_n_bufs or DEFAULT_MAX_N_BUFS)
    return deserialize_graph(byte_arr)
