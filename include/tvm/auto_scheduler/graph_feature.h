/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/graph_feature.h
 * \brief Feature extraction for the graph cost model
 */

#ifndef TVM_AUTO_GRAPH_FEATURE_H_
#define TVM_AUTO_GRAPH_FEATURE_H_

#include <string>
#include <vector>
#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/measure.h>

namespace tvm {
namespace auto_scheduler {

static const int NODE_FEATURE_LENGTH = 1324;
static const int EDGE_FEATURE_LENGTH = 4;
struct Edge {
  int src;
  int dst;
  float feature[EDGE_FEATURE_LENGTH];
};

// {"stmtnode": 0, ""}
struct Node {
  int node_type;
  int id;
  float feature[NODE_FEATURE_LENGTH];
};

void GetGraph(const State& state,
                     const SearchTask& task,
                     int max_n_bufs,
                     std::vector<Node>* node_list,
                     std::vector<Edge>* edge_list);


/*! \brief Get PerStmt feature from states and different tasks */
void GetGraphFromStates(const Array<State>& states,
                        const std::vector<SearchTask>& tasks,
                        int max_n_bufs,
                        std::vector<std::vector<Node> >* node_list,
                        std::vector<std::vector<Edge> >* edge_list);


/*! \brief Get graph from states and the same task */
void GetGraphFromStates(const Array<State>& states,
                        const SearchTask task,
                        int max_n_bufs,
                        std::vector<std::vector<Node> >* node_list,
                        std::vector<std::vector<Edge> >* edge_list);


/*! \brief Get graph from a log file */
void GetGraphFromFile(const std::string& filename,
                      int n_lines,
                      int max_n_bufs,
                      std::vector<std::vector<Node> >* node_list,
                      std::vector<std::vector<Edge> >* edge_list,
                      std::vector<float>* normalized_throughputs,
                      std::vector<int>* task_ids);


/*! \brief Get graph from measure pairs */
void GetGraphFromMeasurePairs(const Array<MeasureInput>& inputs,
                              const Array<MeasureResult>& results,
                              int skip_first_n_feature_extraction,
                              int max_n_bufs,
                              std::vector<std::vector<Node> >* node_list,
                              std::vector<std::vector<Edge> >* edge_list,
                              std::vector<float>* normalized_throughputs,
                              std::vector<int>* task_ids);


}   // namespace auto_scheduler
}   // namespace tvm

#endif  // TVM_AUTO_GRAPH_FEATURE_H_
