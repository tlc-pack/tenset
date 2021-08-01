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
 * \file auto_scheduler/feature.h
 * \brief Feature extraction for the cost model.
 * We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
 * so we call this feature as "per-store" feature.
 * The cost model also does prediction for each BufferStoreNode statement and aggregates
 * the predictions as the whole score for a TVM IR (Stmt).
 *
 * The feature specification is defined by `src/auto_scheduler/feature.cc:: FeatureSet`
 */

#ifndef TVM_AUTO_SCHEDULER_FEATURE_H_
#define TVM_AUTO_SCHEDULER_FEATURE_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/tir/op_attr_types.h>

#include <string>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace auto_scheduler {

/*!
 * \brief Get per-store feature from a TIR Stmt
 * \param stmt The input lowered TIR statement
 * \param cache_line_size The size of cache line in bytes
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned feature vector
 */
void GetPerStoreFeature(const Stmt& stmt, int cache_line_size, int max_n_bufs,
                        std::vector<float>* ret);

/*
 * \brief Get the names of elements in the feature vector. Use this for debug and inspection.
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned names.
 */
void GetPerStoreFeatureName(int max_n_bufs, std::vector<std::string>* ret);

/*!
 * \brief Get per-store feature from states of the same task
 * \param states The input states
 * \param task The same search task for all states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const SearchTask& task,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float> >* features);

/*!
 * \brief Get per-store feature from states of different tasks
 * \param states The input states
 * \param tasks The search tasks corresponding to the input states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const std::vector<SearchTask>& tasks,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float> >* features);

/*!
 * \brief Get per-store features from a log file
 * \param filename The name of log file
 * \param max_lines Only read the first n lines of the file
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromFile(const std::string& filename, int max_lines, int max_n_bufs,
                                 std::vector<std::vector<float> >* features,
                                 std::vector<float>* normalized_throughputs,
                                 std::vector<int>* task_ids);

/*!
 * \brief Get per-store features from measurement input/result pairs
 * \param inputs The measurement inputs
 * \param results The measurement results
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n measurement pairs
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                         const Array<MeasureResult>& results,
                                         int skip_first_n_feature_extraction, int max_n_bufs,
                                         std::vector<std::vector<float> >* features,
                                         std::vector<float>* normalized_throughputs,
                                         std::vector<int>* task_ids);

template <class T>
using BufferMap = std::unordered_map<Buffer, T, ObjectHash, ObjectEqual>;

// Data reuse type
enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };

// Buffer access type
enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };

struct BufferAccess {
  // data reuse type
  BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  std::vector<std::vector<PrimExpr>> indices;
};

struct BufferAccessFeature {
  std::string buffer_name;        // The name of the buffer
  BufferAccessType acc_type;      // The type of the access
  float bytes;                    // The touched memory in bytes
  float unique_bytes;             // The touched unique memory in bytes
  float lines;                    // The number of touched cache lines
  float unique_lines;             // The number touched unique cache lines
  ReuseType reuse_type;           // Tye type of data reuse
  float reuse_dis_iter;           // The reuse distance in iterator number
  float reuse_dis_bytes;          // The reuse distance in total touched bytes
  float reuse_ct;                 // The reuse ratio
  float bytes_d_reuse_ct;         // bytes / reuse_ct
  float unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
  float lines_d_reuse_ct;         // lines / reuse_ct
  float unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
  float stride;                   // The stride in access
};

inline float slog(float x);

int64_t GetLoopExtent(const ForNode* node);

std::tuple<ReuseType, float, float, float> ComputeReuse(
                                        const Buffer& buf,
                                        const std::vector<std::vector<PrimExpr> >& indices,
                                        const std::vector<const ForNode*>& for_loop_stack,
                                        const std::unordered_map<const ForNode*, BufferMap<std::vector<
                                            std::tuple<BufferAccessType, int64_t, int> > > >& for_touch_regions);

void ComputeRegion(
    const std::vector<std::vector<PrimExpr> > &indices,
    arith::Analyzer* ana,
    std::vector<int>* region);

int64_t ComputeStride(const std::vector<std::vector<PrimExpr> >& indices,
                      const std::vector<int>& shape,
                      const VarNode* stride_var);

class BufferAccessExtractor {
 public:
  void ExtractReads(const PrimExpr& expr);

  void InsertAccess(const Buffer& buf, BufferAccessType acc_type, const Array<PrimExpr>& indices);

  BufferMap<BufferAccess> buf_accesses;
};

class MathOpCounter {
 public:
  size_t float_mad;         // The number of float MAD (Multiply–add) ops
  size_t float_addsub;      // The number of float add and sub ops
  size_t float_mul;         // The number of float multiply ops
  size_t float_divmod;      // The number of float div and mod ops
  size_t float_cmp;         // The number of float comparison ops
  size_t float_math_func;   // The number of float math func calls
  size_t float_other_func;  // The number of other float func calls
  size_t int_mad;           // The number of integer MAD (Multiply–add) ops
  size_t int_addsub;        // The number of integer add and sub ops
  size_t int_mul;           // The number of float multiply ops
  size_t int_divmod;        // The number of float div and mod ops
  size_t int_cmp;           // The number of float comparison ops
  size_t int_math_func;     // The number of float math func calls
  size_t int_other_func;    // The number of other float func calls
  size_t bool_op;           // The number of bool ops
  size_t select_op;         // The number of select ops

  OpAttrMap<TCallEffectKind> op_call_effect_;
};


}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_FEATURE_H_
