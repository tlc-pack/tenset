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
 * \file auto_scheduler/feature.cc
 * \brief Feature extraction for the cost model
 */

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/graph_feature.h>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <chrono>

#include "search_policy/utils.h"
#include "utils.h"

namespace tvm {
/* Import the function from driver_api.cc */
extern void GetBinds(const Array<te::Tensor>& args, bool compact,
                     const std::unordered_map<te::Tensor, tir::Buffer>& binds,
                     Map<te::Tensor, tir::Buffer>* out_binds, Array<ObjectRef>* out_arg_list);
}  // namespace tvm


namespace tvm {
namespace auto_scheduler {

using namespace tvm::tir;
using arith::ConstIntBound;
using arith::Analyzer;

template <class T>
using BufferMap = std::unordered_map<Buffer, T, ObjectHash, ObjectEqual>;

// The number of samples to extract for arithmetic intensity curves
static const int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;

// Annotation position encoding
enum class AnnotationPosType : int {
  kPosNone = 0,           // Does not have this kind of annotation
  kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
  kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
  kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
  kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
  kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
  kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
  kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
};

// Buffer access type
// enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };

// Accesses to a buffer
//struct BufferAccess {
  // data reuse type
  //BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  //std::vector<std::vector<PrimExpr>> indices;
//};

// Data reuse type
// enum class ReuseType : int { kLoopMultipleRead = 0, kSerialMultipleReadWrite = 1, kNoReuse = 2 };

// Feature for an access of a buffer
/*
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
*/
// Feature set of a BufferStore statement
struct FeatureSet {
  // Group 1: Computation related features
  float float_mad;                  // The number of float MAD (Multiply–add) ops
  float float_addsub;               // The number of float add and sub ops
  float float_mul;                  // The number of float multiply ops
  float float_divmod;               // The number of float div and mod ops
  float float_cmp;                  // The number of float comparison ops
  float float_math_func;            // The number of float math func calls
  float float_other_func;           // The number of other float func calls
  float int_mad;                    // The number of integer MAD (Multiply–add) ops
  float int_addsub;                 // The number of integer add and sub ops
  float int_mul;                    // The number of float multiply ops
  float int_divmod;                 // The number of float div and mod ops
  float int_cmp;                    // The number of float comparison ops
  float int_math_func;              // The number of float math func calls
  float int_other_func;             // The number of other float func calls
  float bool_op;                    // The number of bool ops
  float select_op;                  // The number of select ops
  float vec_num;                    // The number of vectorized iterators
  float vec_prod;                   // The product of the lengths of vectorized iterators
  float vec_len;                    // The length of the innermost vectorized iterator
  AnnotationPosType vec_type;       // The type of vectorization position
  float unroll_num;                 // The number of unrolled iterators
  float unroll_prod;                // The product of the lengths of vectorized iterators
  float unroll_len;                 // The length of the innermost unrolled iterator
  AnnotationPosType unroll_type;    // The type of unroll position
  float parallel_num;               // The number of paralleled iterators
  float parallel_prod;              // The product of the lengths of paralleled iterators
  float parallel_len;               // The length of the innermost paralleled iterators
  AnnotationPosType parallel_type;  // The type of parallel position
  float is_gpu;                     // Whether it is a GPU task
  float blockIdx_x_len;             // The length of blockIdx.x
  float blockIdx_y_len;             // The length of blockIdx.y
  float blockIdx_z_len;             // The length of blockIdx.z
  float threadIdx_x_len;            // The length of threadIdx.x
  float threadIdx_y_len;            // The length of threadIdx.y
  float threadIdx_z_len;            // The length of threadIdx.z
  float vthread_len;                // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  std::vector<BufferAccessFeature> access_feas;

  // Group 3: Arithmetic intensity related features
  float arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];  // points sampled from the
                                                                // arithmetic intensity curve

  // Group 4: Allocation related features
  float alloc_size;        // The size of allocated buffer in bytes
  float alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
  float alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
  float alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  float outer_prod;            // The product of lengths of outer loops
  float num_loops;             // The number of outer loops
  float auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};

// Return the min of a for loop
int64_t GetLoopMin(const ForNode* node) {
  auto pint = node->min.as<IntImmNode>();
  if (pint != nullptr) {
    return pint->value;
  } else {
    return 0;
  }
}

inline float slog2(float x) {
  return slog(slog(x));
}


class NodeGather : public StmtExprVisitor {
 public:

  void VisitStmt(const Stmt& n) {
    int primExprType = 0;
    int64_t forExtent = 0;
    int64_t forMin = 0;
    int forKind = 0;

    if (n->IsInstance<LetStmtNode>()) {
      primExprType = 0;
    } else if (n->IsInstance<AttrStmtNode>()) {
      primExprType = 1;
    } else if (n->IsInstance<IfThenElseNode>()) {
      primExprType = 2;
    } else if (n->IsInstance<ForNode>()) {
      primExprType = 3;
      forExtent = GetLoopExtent(n.as<ForNode>());
      forMin = GetLoopMin(n.as<ForNode>());
      forKind = (int)n.as<ForNode>()->kind;
    } else if (n->IsInstance<AllocateNode>()) {
      primExprType = 4;
    } else if (n->IsInstance<StoreNode>()) {
      primExprType = 5;
    } else if (n->IsInstance<BufferStoreNode>()) {
      primExprType = 6;
    } else if (n->IsInstance<BufferRealizeNode>()) {
      primExprType = 7;
    } else if (n->IsInstance<AssertStmtNode>()) {
      primExprType = 8;
    } else if (n->IsInstance<ProducerStoreNode>()) {
      primExprType = 9;
    } else if (n->IsInstance<ProducerRealizeNode>()) {
      primExprType = 10;
    } else if (n->IsInstance<PrefetchNode>()) {
      primExprType = 11;
    } else if (n->IsInstance<SeqStmtNode>()) {
      primExprType = 12;
    } else if (n->IsInstance<EvaluateNode>()) {
      primExprType = 13;
    }

    if (node_to_index.find(n.get()) == node_to_index.end()) {
      Node newNode = {
          0,
          count,
          {0}
      };
      newNode.feature[0] = 0;
      newNode.feature[primExprType+1] = 1;
      if (primExprType == 3) {
        
        newNode.feature[35] = slog2(forExtent);
        newNode.feature[36] = slog2(forMin);
        newNode.feature[37+forKind] = 1;

        const ForNode* node = n.as<ForNode>();

        int64_t loop_extent = GetLoopExtent(node);
        if (node->kind == ForKind::kVectorized) {
          vec_for_stack.push_back(node);
        } else if (node->kind == ForKind::kUnrolled) {
          unroll_for_stack.push_back(node);
        } else if (node->kind == ForKind::kParallel) {
          parallel_for_stack.push_back(node);
        }

        outer_loop_prod *= loop_extent;
        for_loop_stack.push_back(node);

        node_list.push_back(newNode);
        node_to_index[n.get()] = count;
        count++;

        StmtExprVisitor::VisitStmt_(node);
        for_loop_stack.pop_back();
        outer_loop_prod /= loop_extent;

        if (node->kind == ForKind::kVectorized) {
          vec_for_stack.pop_back();
        } else if (node->kind == ForKind::kUnrolled) {
          unroll_for_stack.pop_back();
        } else if (node->kind == ForKind::kParallel) {
          parallel_for_stack.pop_back();
        }

      }

      if (primExprType == 6) {

        const BufferStoreNode* node = n.as<BufferStoreNode>();
         
        MathOpCounter math_op_counter;
        math_op_counter(node->value);
        std::vector<float> mem_bytes_list;
        std::vector<float> compute_ops_list;
        double cur_compute_ops;
        
        FeatureSet feature;
        // Group 1: Computation related features
        ExtractComputationFeature(node, math_op_counter, feature);

        // Group 2: Buffer access related features (per buffer)
        ExtractBufferAccessFeature(node, math_op_counter, &cur_compute_ops, &compute_ops_list,
                                  &mem_bytes_list, feature);

        // Group 3: Arithmetic intensity related features
        ExtractArithmeticIntensityFeature(node, cur_compute_ops, compute_ops_list, mem_bytes_list, feature);

        // Group 4: Allocation related features
        ExtractOuterScopeFeature(node, feature);

        std::vector<std::pair<float, float> > buf_order_key;
        for (const auto& acc_fea : feature.access_feas) {
          buf_order_key.emplace_back(acc_fea.lines, acc_fea.bytes);
        }
        std::vector<int> buf_order(buf_order_key.size());
        std::iota(buf_order.begin(), buf_order.end(), 0);

        auto cmp = [&buf_order_key](int l, int r) {
          return buf_order_key[l].first > buf_order_key[r].first
              || (buf_order_key[l].first == buf_order_key[r].first
                  && buf_order_key[l].second > buf_order_key[r].second);
        };
        std::sort(buf_order.begin(), buf_order.end(), cmp);
        int n_bufs = std::min(max_n_bufs, static_cast<int>(buf_order.size()));
        buf_order.resize(n_bufs);


        int i = 42;

        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        //vectype
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        //unroll_type
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        //parallel_type
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);
        newNode.feature[i++] = slog2(feature.float_mad);

        for (int idx = 0; idx < ARITH_INTENSITY_CURVE_SAMPLE_N; idx++){
          newNode.feature[i++] = slog2(feature.arith_intensity_curve[idx]);
        }

        newNode.feature[i++] = slog2(feature.alloc_size);
        newNode.feature[i++] = slog2(feature.alloc_outer_prod);
        newNode.feature[i++] = slog2(feature.alloc_inner_prod);
        newNode.feature[i++] = slog2(feature.alloc_prod);

        newNode.feature[i++] = slog2(feature.outer_prod);
        newNode.feature[i++] = slog2(feature.num_loops);
        newNode.feature[i++] = slog2(feature.auto_unroll_max_step);

        for (int idx : buf_order) {
          const auto& acc_fea = feature.access_feas[idx];
          for (int j = 0; j <= (int)BufferAccessType::kReadWrite; ++j) {
            newNode.feature[i++] = (j == (int)acc_fea.acc_type);
          }
          newNode.feature[i++] = slog2(acc_fea.bytes);
          newNode.feature[i++] = slog2(acc_fea.unique_bytes);
          newNode.feature[i++] = slog2(acc_fea.lines);
          newNode.feature[i++] = slog2(acc_fea.unique_lines);
          for (int j = 0; j <= (int)ReuseType::kNoReuse; ++j) {
            newNode.feature[i++] = ((int)acc_fea.reuse_type == j);
          }

          newNode.feature[i++] = slog2(acc_fea.reuse_dis_iter);
          newNode.feature[i++] = slog2(acc_fea.reuse_dis_bytes);
          newNode.feature[i++] = slog2(acc_fea.reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.bytes_d_reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.unique_bytes_d_reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.lines_d_reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.unique_lines_d_reuse_ct);
          newNode.feature[i++] = slog2(acc_fea.stride);

        }
        // - fill padding
        for (int k = 0; k < max_n_bufs - n_bufs; ++k) {
          for (int j = 0; j <= (int)BufferAccessType::kReadWrite; ++j) {  // 3
            newNode.feature[i++] = 0.0f;
          }
          newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f;
          for (int j = 0; j <= (int)ReuseType::kNoReuse; ++j) {   // 3
            newNode.feature[i++] = 0.0f;
          }
          newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f;
          newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f; newNode.feature[i++] = 0.0f;
        }

        std::cout << "i: " << i << std::endl;

        }
          if (primExprType != 3) {
              node_list.push_back(newNode);
              node_to_index[n.get()] = count;
              count++;
          }
      }
    if (primExprType != 3) {
        StmtExprVisitor::VisitStmt(n);
    }
  }

  void VisitExpr(const PrimExpr& n) {
    //    int count = static_cast<int>(node_to_index.size());
    int primExprType = 0;

    if (n->IsInstance<CallNode>()) {
      primExprType = 0;
    } else if (n->IsInstance<AddNode>()) {
      primExprType = 1;
    } else if (n->IsInstance<SubNode>()) {
      primExprType = 2;
    } else if (n->IsInstance<MulNode>()) {
      primExprType = 3;
    } else if (n->IsInstance<DivNode>()) {
      primExprType = 4;
    } else if (n->IsInstance<ModNode>()) {
      primExprType = 5;
    } else if (n->IsInstance<FloorDivNode>()) {
      primExprType = 6;
    } else if (n->IsInstance<FloorModNode>()) {
      primExprType = 7;
    } else if (n->IsInstance<MinNode>()) {
      primExprType = 8;
    } else if (n->IsInstance<MaxNode>()) {
      primExprType = 9;
    } else if (n->IsInstance<EQNode>()) {
      primExprType = 10;
    } else if (n->IsInstance<NENode>()) {
      primExprType = 11;
    } else if (n->IsInstance<LTNode>()) {
      primExprType = 12;
    } else if (n->IsInstance<LENode>()) {
      primExprType = 13;
    } else if (n->IsInstance<GTNode>()) {
      primExprType = 14;
    } else if (n->IsInstance<GENode>()) {
      primExprType = 15;
    } else if (n->IsInstance<AndNode>()) {
      primExprType = 16;
    } else if (n->IsInstance<OrNode>()) {
      primExprType = 17;
    } else if (n->IsInstance<ReduceNode>()) {
      primExprType = 18;
    } else if (n->IsInstance<CastNode>()) {
      primExprType = 19;
    } else if (n->IsInstance<NotNode>()) {
      primExprType = 20;
    } else if (n->IsInstance<SelectNode>()) {
      primExprType = 21;
    } else if (n->IsInstance<RampNode>()) {
      primExprType = 22;
    } else if (n->IsInstance<BroadcastNode>()) {
      primExprType = 23;
    } else if (n->IsInstance<ShuffleNode>()) {
      primExprType = 24;
    } else if (n->IsInstance<IntImmNode>()) {
      primExprType = 25;
    } else if (n->IsInstance<FloatImmNode>()) {
      primExprType = 26;
    } else if (n->IsInstance<StringImmNode>()) {
      primExprType = 27;
    } else if (n->IsInstance<VarNode>()) {
      primExprType = 28;
    } else if (n->IsInstance<SizeVarNode>()) {
      primExprType = 29;
    } else if (n->IsInstance<BufferLoadNode>()) {
      primExprType = 30;
    } else if (n->IsInstance<ProducerLoadNode>()) {
      primExprType = 31;
    } else if (n->IsInstance<LoadNode>()) {
      primExprType = 32;
    } else if (n->IsInstance<LetNode>()) {
      primExprType = 33;
    }

    if (node_to_index.find(n.get()) == node_to_index.end()) {
      Node newNode = {
          1,
          count,
          {0}
      };
      newNode.feature[0] = 1;
      newNode.feature[primExprType+1] = 1;

      if (primExprType == 25) {
        newNode.feature[41] = slog2(n.as<IntImmNode>()->value);
      }
      if (primExprType == 26) {
        newNode.feature[41] = slog2(n.as<FloatImmNode>()->value);
      }
      node_list.push_back(newNode);
      node_to_index[n.get()] = count;
      count++;
    }

    StmtExprVisitor::VisitExpr(n);
  }

  // Extract computation related features (group 1)
  void ExtractComputationFeature(const BufferStoreNode* node,
                                 const MathOpCounter& math_op_counter, FeatureSet& fea) {

    // Computation related features
    fea.float_mad = outer_loop_prod_ * math_op_counter.float_mad;
    fea.float_addsub = outer_loop_prod_ * math_op_counter.float_addsub;
    fea.float_mul = outer_loop_prod_ * math_op_counter.float_mul;
    fea.float_divmod = outer_loop_prod_ * math_op_counter.float_divmod;
    fea.float_cmp = outer_loop_prod_ * math_op_counter.float_cmp;
    fea.float_math_func = outer_loop_prod_ * math_op_counter.float_math_func;
    fea.float_other_func = outer_loop_prod_ * math_op_counter.float_other_func;
    fea.int_mad = outer_loop_prod_ * math_op_counter.int_mad;
    fea.int_addsub = outer_loop_prod_ * math_op_counter.int_addsub;
    fea.int_mul = outer_loop_prod_ * math_op_counter.int_mul;
    fea.int_divmod = outer_loop_prod_ * math_op_counter.int_divmod;
    fea.int_math_func = outer_loop_prod_ * math_op_counter.int_math_func;
    fea.int_cmp = outer_loop_prod_ * math_op_counter.int_cmp;
    fea.int_other_func = outer_loop_prod_ * math_op_counter.int_other_func;
    fea.bool_op = outer_loop_prod_ * math_op_counter.bool_op;
    fea.select_op = outer_loop_prod_ * math_op_counter.select_op;

    fea.vec_len = fea.unroll_len = fea.parallel_len = 0.0f;
    fea.vec_type = fea.unroll_type = fea.parallel_type = AnnotationPosType::kPosNone;

    fea.vec_num = vec_for_stack_.size();
    if (!vec_for_stack_.empty()) {
      fea.vec_len = GetLoopExtent(vec_for_stack_.back());
      fea.vec_prod = 1.0;
      for (const ForNode* pfor : vec_for_stack_) {
        fea.vec_prod *= GetLoopExtent(pfor);
      }
      fea.vec_type = AnnotationPosType::kPosMixed;
      // todo(merrymercy): this feature requires operation (tvm.compute) information
      // GetAnnotationPosEncoding(vec_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.unroll_num = unroll_for_stack_.size();
    if (!unroll_for_stack_.empty()) {
      fea.unroll_len = GetLoopExtent(unroll_for_stack_.back());
      fea.unroll_prod = 1.0;
      for (const ForNode* pfor : unroll_for_stack_) {
        fea.unroll_prod *= GetLoopExtent(pfor);
      }
      fea.unroll_type = AnnotationPosType::kPosMixed;
      // GetAnnotationPosEncoding(unroll_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    fea.parallel_num = parallel_for_stack_.size();
    if (!parallel_for_stack_.empty()) {
      fea.parallel_len = GetLoopExtent(parallel_for_stack_.back());
      fea.parallel_prod = 1.0;
      for (const ForNode* pfor : parallel_for_stack_) {
        fea.parallel_prod *= GetLoopExtent(pfor);
      }
      fea.parallel_type = AnnotationPosType::kPosMixed;
      // GetAnnotationPosEncoding(parallel_for_stack_.back()->loop_var,
      // node->args, pcompute->axis, pcompute->reduce_axis);
    }

    // GPU threads
    fea.is_gpu = is_gpu_;
    fea.blockIdx_x_len = blockIdx_x_len_;
    fea.blockIdx_y_len = block_idx_y_len_;
    fea.blockIdx_z_len = block_idx_z_len_;
    fea.threadIdx_x_len = threadIdx_x_len_;
    fea.threadIdx_y_len = thread_idx_y_len_;
    fea.threadIdx_z_len = thread_idx_z_len_;
    fea.vthread_len = vthread_len_;
  }

  // Extract buffer access related features (group 2)
  void ExtractBufferAccessFeature(const BufferStoreNode* node, const MathOpCounter& math_op_counter,
                                  double* cur_compute_ops, std::vector<float>* compute_ops_list,
                                  std::vector<float>* mem_bytes_list, FeatureSet& fea) {

    // Extract all buffer accesses
    std::vector<BufferAccessFeature> acc_feas;
    BufferAccessExtractor buf_extractor;
    buf_extractor.InsertAccess(node->buffer, BufferAccessType::kWrite, node->indices);
    buf_extractor.ExtractReads(node->value);

    // Compute touched region for all outer loops
    for (auto x : for_loop_stack_) {
      ana_.Bind(x->loop_var, Range::FromMinExtent(x->min, 1), true);
    }

    mem_bytes_list->reserve(for_loop_stack_.size());
    compute_ops_list->reserve(for_loop_stack_.size());

    *cur_compute_ops = math_op_counter.float_mad + math_op_counter.float_addsub +
                       math_op_counter.float_mul + math_op_counter.float_divmod +
                       math_op_counter.float_cmp + math_op_counter.float_math_func +
                       math_op_counter.float_other_func;

    std::vector<int> tmp_region;
    for (int i = static_cast<int>(for_loop_stack_.size()) - 1; i >= 0; i--) {
      const ForNode* p_for = for_loop_stack_[i];

      ana_.Bind(p_for->loop_var,
                Range::FromMinExtent(for_loop_stack_[i]->min, for_loop_stack_[i]->extent), true);

      // Note, here we do overwrite.
      // So if there are multiple BufferStoreNode, the last one will overwrite the first few.
      // e.g. The update part in gemm will overwrite the init part.
      BufferMap<std::vector<std::tuple<BufferAccessType, int64_t, int>>>& buffer_regions_map =
          for_touch_regions_[p_for];

      int64_t mem_bytes = 0;
      for (const auto& x : buf_extractor.buf_accesses) {
        const Buffer& t = x.first;
        const BufferAccess& acc = x.second;

        ComputeRegion(acc.indices, &ana_, &tmp_region);
        int64_t touched_size = ElementProduct(tmp_region);
        buffer_regions_map[t].push_back(
            std::make_tuple(acc.acc_type, touched_size, t->dtype.bytes()));
        mem_bytes += touched_size * t->dtype.bytes();
      }

      mem_bytes_list->push_back(std::log2(mem_bytes));
      *cur_compute_ops *= GetLoopExtent(for_loop_stack_[i]);
      compute_ops_list->push_back(std::log2(*cur_compute_ops));
    }

    //  Buffer access related features (per buffer)
    for (const auto& x : buf_extractor.buf_accesses) {
      const Buffer& t = x.first;
      const BufferAccess& acc = x.second;

      std::vector<int> int_shape;
      for (const auto& dim : t->shape) {
        int_shape.push_back(GetIntImm(dim));
      }

      size_t ele_bytes = t->dtype.bytes();

      // calculate bytes
      float bytes = outer_loop_prod_ * ele_bytes;
      float unique_bytes;

      // calculate cache lines
      int64_t stride;
      float lines;
      float unique_lines;

      if (for_loop_stack_.empty()) {
        unique_bytes = ele_bytes;
        stride = 0;
        lines = 1.0f;
        unique_lines = 1.0f;
      } else {
        unique_bytes =
            std::get<1>(for_touch_regions_[for_loop_stack_.front()][t].front()) * ele_bytes;

        stride = 0;
        int64_t reduce_ratio = 1;

        int i;
        for (i = static_cast<int>(for_loop_stack_.size()) - 1; i >= 0; i--) {
          stride = ComputeStride(acc.indices, int_shape, for_loop_stack_[i]->loop_var.get());
          if (stride != 0) {
            break;
          }
          reduce_ratio *= GetLoopExtent(for_loop_stack_.back());
        }

        lines = outer_loop_prod_ / reduce_ratio *
                std::min(1.0f, 1.0f * stride * ele_bytes / cache_line_size_);
        lines = std::max(lines, 1.0f);

        // convert `stride` back to the stride of the innermost iterator
        stride = (i == static_cast<int>(for_loop_stack_.size()) - 1 ? stride : 0);

        float n_continuous = ele_bytes;
        for (int i = std::min(static_cast<int>(tmp_region.size()) - 1,
                              static_cast<int>(int_shape.size()) - 1);
             i >= 0; i--) {
          if (tmp_region[i] == int_shape[i]) {
            n_continuous *= tmp_region[i];
            break;
          }
        }
        unique_lines = unique_bytes / std::min(n_continuous, static_cast<float>(cache_line_size_));
        unique_lines = std::max(unique_lines, 1.0f);
      }

      ReuseType reuse_type;
      float reuse_dis_iter, reuse_dis_bytes, reuse_ct;
      std::tie(reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct) =
          ComputeReuse(t, acc.indices, for_loop_stack_, for_touch_regions_);

      acc_feas.emplace_back();
      BufferAccessFeature& acc_fea = acc_feas.back();

      acc_fea.buffer_name = t->name;
      acc_fea.acc_type = acc.acc_type;
      acc_fea.stride = stride;
      acc_fea.bytes = bytes;
      acc_fea.unique_bytes = unique_bytes;
      acc_fea.lines = lines;
      acc_fea.unique_lines = unique_lines;
      acc_fea.reuse_type = reuse_type;
      acc_fea.reuse_dis_iter = reuse_dis_iter;
      acc_fea.reuse_dis_bytes = reuse_dis_bytes;
      acc_fea.reuse_ct = reuse_ct;
      if (acc_fea.reuse_ct > 0.5) {
        acc_fea.bytes_d_reuse_ct = bytes / reuse_ct;
        acc_fea.unique_bytes_d_reuse_ct = unique_bytes / reuse_ct;
        acc_fea.lines_d_reuse_ct = lines / reuse_ct;
        acc_fea.unique_lines_d_reuse_ct = unique_lines / reuse_ct;
      } else {
        // no reuse, multiply by a magic number '2'
        acc_fea.bytes_d_reuse_ct = bytes * 2;
        acc_fea.unique_bytes_d_reuse_ct = unique_bytes * 2;
        acc_fea.lines_d_reuse_ct = lines * 2;
        acc_fea.unique_lines_d_reuse_ct = unique_lines * 2;
      }
    }

    fea.access_feas = acc_feas;
  }

  // Extract arithmetic intensity related feature (group 3)
  void ExtractArithmeticIntensityFeature(const BufferStoreNode* node, double cur_compute_ops,
                                         const std::vector<float>& compute_ops_list,
                                         const std::vector<float>& mem_bytes_list,
                                         FeatureSet& fea) {

    // Compute arithmetic intensity curve (y axis : arithmetic intensity, x axis : flops).
    // We use piecewise linear interpolation to fit this curve.
    int pt = 0;
    if (cur_compute_ops <= 0 || compute_ops_list.empty()) {
      std::fill(fea.arith_intensity_curve,
                fea.arith_intensity_curve + ARITH_INTENSITY_CURVE_SAMPLE_N, 0.0);
    } else {
      for (size_t i = 0; i < ARITH_INTENSITY_CURVE_SAMPLE_N; ++i) {
        float cur_compute_ops = compute_ops_list.back() * (i + 1) / ARITH_INTENSITY_CURVE_SAMPLE_N;
        while (compute_ops_list[pt] < cur_compute_ops - 1e-4) {
          pt++;
        }
        ICHECK_LT(pt, compute_ops_list.size());

        float value;
        if (pt == 0) {
          value = compute_ops_list[pt] / mem_bytes_list[pt];
        } else {
          float base = compute_ops_list[pt - 1] / mem_bytes_list[pt - 1];
          float slope = (compute_ops_list[pt] / mem_bytes_list[pt] -
                         compute_ops_list[pt - 1] / mem_bytes_list[pt - 1]) /
                        (compute_ops_list[pt] - compute_ops_list[pt - 1]);
          value = base + slope * (cur_compute_ops - compute_ops_list[pt - 1]);
        }
        fea.arith_intensity_curve[i] = value;
      }
    }
  }

  // Extract allocation related features (group 4)
  void ExtractAllocationFeature(const BufferRealizeNode* node, FeatureSet& fea) {

    float allocation_size = 1.0f;
    for (const auto& x : node->bounds) {
      allocation_size *= GetIntImm(x->extent);
    }
    // allocation feature
    fea.alloc_size = allocation_size * node->buffer->dtype.bytes();
    fea.alloc_prod = allocation_size * outer_loop_prod_;
    fea.alloc_outer_prod = outer_loop_prod_;
    fea.alloc_inner_prod = fea.outer_prod / outer_loop_prod_;
  }

  // Extract outer scope related features (group 5)
  void ExtractOuterScopeFeature(const BufferStoreNode* node, FeatureSet& fea) {

    fea.outer_prod = outer_loop_prod_;
    fea.num_loops = for_loop_stack_.size();
    fea.auto_unroll_max_step = cur_auto_unroll_max_step_;
  }

  std::unordered_map<const Object*, int> node_to_index;
  std::vector<Node> node_list;
  int count = 0;
  int max_n_bufs = 3;

  std::unordered_map<std::string, int> buffer_lookup;
  int buffer_count = 0;

  float outer_loop_prod = 1.0f;

  std::vector<const ForNode*> for_loop_stack;
  std::vector<const ForNode*> parallel_for_stack;
  std::vector<const ForNode*> vec_for_stack;
  std::vector<const ForNode*> unroll_for_stack;

  std::unordered_map<const ForNode*, BufferMap<std::vector<
  std::tuple<BufferAccessType, int64_t, int> > > > for_touch_regions;
  private:
    const int cache_line_size_ = 64;
    // The product of outer loop
    float outer_loop_prod_ = 1.0f;
    // GPU-related features
    bool is_gpu_{false};
    int blockIdx_x_len_{1};
    int block_idx_y_len_{1};
    int block_idx_z_len_{1};
    int threadIdx_x_len_{1};
    int thread_idx_y_len_{1};
    int thread_idx_z_len_{1};
    int vthread_len_{1};
    int16_t cur_auto_unroll_max_step_{0};

};


class EdgeGather: public StmtExprVisitor {
 public:
  void VisitStmt_(const AttrStmtNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->value.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const ForNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->min.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[2] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->loop_var.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);

    dst_idx = node_to_index[node->extent.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[2] = 1;
    edge_list.push_back(newEdge);


    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const BufferStoreNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->value.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    for (std::size_t i = 0; i < node->indices.size(); i++) {
      dst_idx = node_to_index[node->indices[i].get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }

    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const BufferRealizeNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = node_to_index[node->condition.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[node->body.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    for (const Range &range : node->bounds) {
      dst_idx = node_to_index[range->min.get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
      dst_idx = node_to_index[range->extent.get()];
      newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }

    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const SeqStmtNode* node) {
    int src_idx = node_to_index[static_cast<const Object*>(node)];
    int dst_idx = 0;
    for (const Stmt &s : node->seq) {
      dst_idx = node_to_index[s.get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = 0;
    for (const PrimExpr &primExpr : op->indices) {
      dst_idx = node_to_index[primExpr.get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const AddNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const DivNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ModNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorDivNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorModNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MinNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MaxNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const EQNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const NENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LTNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const LENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const GTNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const GENode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const AndNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const OrNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx = node_to_index[op->a.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    dst_idx = node_to_index[op->b.get()];
    newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    newEdge.feature[3] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ReduceNode* op) {
    int src_idx = node_to_index[static_cast<const Object*>(op)];
    int dst_idx;
    for (int i = 0; op->source.size(); i++) {
      dst_idx = node_to_index[op->source[i].get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    for (int i = 0; op->axis.size(); i++) {
      dst_idx = node_to_index[op->axis[i].get()];
      Edge newEdge = {
          src_idx,
          dst_idx,
          {0}
      };
      newEdge.feature[0] = 1;
      edge_list.push_back(newEdge);
    }
    dst_idx = node_to_index[op->condition.get()];
    Edge newEdge = {
        src_idx,
        dst_idx,
        {0}
    };
    newEdge.feature[0] = 1;
    edge_list.push_back(newEdge);
    StmtExprVisitor::VisitExpr_(op);
  }

  std::vector<Edge> edge_list;
  std::unordered_map<const Object*, int> node_to_index;

};


TVMByteArray SerializeGraph(std::vector<std::vector<Edge> >& edge_list,
                            std::vector<std::vector<Node> >& node_list,
                            std::vector<float>&& normalized_throughputs,
                            std::vector<int>&& task_ids,
                            std::vector<float>&& min_costs,
                            std::vector<char>* out_data) {
  /* Serialization format
   * {
   *   size_vector;
   *   Edge edges[n_edges];
   *   Node nodes[n_nodes];
   *   normalized_throughputs;
   *   task_ids;
   *   min_costs;
   * }
   */

  std::vector<int> size_vector;
  size_t total_bytes = 0;
  int n = edge_list.size();

  //std::cout << "Length of edge list: " << n << std::endl;

  // serialize sizes
  size_t size_vector_size = 1 + n + n + 3;
  total_bytes += size_vector_size * sizeof(int);

  size_vector.reserve(size_vector_size);
  size_vector.push_back(edge_list.size());

  for (const auto& x : edge_list) {
    size_vector.push_back(static_cast<int>(x.size())*2);
    total_bytes += sizeof(Edge) * x.size() * 2;
  }
  for (const auto& x : node_list) {
    size_vector.push_back(static_cast<int>(x.size()));
    total_bytes += sizeof(Node) * x.size();
  }

  size_vector.push_back(static_cast<int>(normalized_throughputs.size()));
  total_bytes += sizeof(float) * normalized_throughputs.size();
  size_vector.push_back(static_cast<int>(task_ids.size()));
  total_bytes += sizeof(int) * task_ids.size();
  size_vector.push_back(static_cast<int>(min_costs.size()));
  total_bytes += sizeof(float) * min_costs.size();

  //std::cout << "Sizes: " << normalized_throughputs.size() << 
  //" " << task_ids.size() <<
  //" " << min_costs.size() <<std::endl;

  CHECK_EQ(size_vector.size(), size_vector_size);

  out_data->reserve(total_bytes);
  char* ptr = out_data->data();

  // serialize size_vector
  memmove(ptr, reinterpret_cast<char*>(size_vector.data()), size_vector.size() * sizeof(int));
  ptr += size_vector.size() * sizeof(int);

  // serialize edge list
  for (auto& x : edge_list) {
    for (Edge& edge : x) {
      memcpy(ptr, &edge.src, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &edge.dst, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, edge.feature, sizeof(float) * EDGE_FEATURE_LENGTH);
      ptr += sizeof(float) * EDGE_FEATURE_LENGTH;
    }
//    x.clear();
    //adding backward edges
    for (Edge& edge : x) {
      memcpy(ptr, &edge.dst, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &edge.src, sizeof(int));
      ptr += sizeof(int);
      edge.feature[0] = 0;
      edge.feature[1] = 1;
      memcpy(ptr, edge.feature, sizeof(float) * EDGE_FEATURE_LENGTH);
      ptr += sizeof(float) * EDGE_FEATURE_LENGTH;
    }
    x.clear();
  }

  // serialize node list
  for (auto& x : node_list) {
    for (Node& node : x) {
      memcpy(ptr, &node.node_type, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, &node.id, sizeof(int));
      ptr += sizeof(int);
      memcpy(ptr, node.feature, sizeof(float) * NODE_FEATURE_LENGTH);
      ptr += sizeof(float) * NODE_FEATURE_LENGTH;
    }
    x.clear();
  }

  // serialize normalized_throughputs
  memmove(ptr, reinterpret_cast<char*>(normalized_throughputs.data()),
          normalized_throughputs.size() * sizeof(float));
  ptr += normalized_throughputs.size() * sizeof(float);

  // serialize task_ids
  memmove(ptr, reinterpret_cast<char*>(task_ids.data()), task_ids.size() * sizeof(int));
  ptr += task_ids.size() * sizeof(int);

  // serialize min_costs
  memmove(ptr, reinterpret_cast<char*>(min_costs.data()), min_costs.size() * sizeof(float));
  ptr += min_costs.size() * sizeof(float);

  CHECK_EQ(ptr - out_data->data(), total_bytes);

  return TVMByteArray{out_data->data(), total_bytes};
}

//int i = 0;
void GetGraph(const State& state, const SearchTask& task, int max_n_bufs,
              std::vector<Node>* node_list, std::vector<Edge>* edge_list,
              std::atomic<int>* error_ct) {
  te::Schedule sch;
  Array<te::Tensor> tensors;

  //LOG(INFO) << "GetGraph start";

  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  
  //LOG(INFO) << "ApplySteps";

  sch = sch.normalize_for_feature_extraction();
  auto bounds = te::InferBound(sch);

  //LOG(INFO) << "bounds";

  // NOTE: Currently, feature extraction with and without layout rewrite
  // returns the same feature vector, so we do not turn on layout rewrite here.
  // In the future, we can improve the feature extraction to reflect this difference.
  try {
    auto stmt = te::ScheduleOps(sch, bounds, false);
    //LOG(INFO) << "ScheduleOps";
    Map<te::Tensor, te::Buffer> out_binds;
    Array<ObjectRef> out_arg_list;
    bool compact = te::VerifyCompactBuffer(stmt);
    const std::string& name = "main";
    GlobalVar global_var(name);

    //LOG(INFO) << "compact";

    // Copied from driver_api.cc::lower
    auto pass_ctx = tvm::transform::PassContext::Current();
    GetBinds(tensors, compact, std::unordered_map<te::Tensor, te::Buffer>(), &out_binds,
             &out_arg_list);
    tir::PrimFunc f = te::SchedulePostProcToPrimFunc(out_arg_list, std::move(stmt), out_binds);
    f = WithAttr(std::move(f), "global_symbol", runtime::String(name));

    //LOG(INFO) << "GetBinds";

    bool noalias = pass_ctx->GetConfig<Bool>("tir.noalias", Bool(true)).value();
    bool disable_vectorize =
        pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    bool instrument_bound_checkers =
        pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();

    if (noalias) {
      f = WithAttr(std::move(f), "tir.noalias", Bool(true));
    }
    auto mod = IRModule(Map<GlobalVar, BaseFunc>({{global_var, f}}));

    if (IsGPUTask(task)) {
      auto pass_list = Array<tvm::transform::Pass>();
      // Phase 0
      pass_list.push_back(tir::transform::InjectPrefetch());
      pass_list.push_back(tir::transform::StorageFlatten(64, instrument_bound_checkers));
      // Phase 1
      pass_list.push_back(tir::transform::NarrowDataType(32));
      pass_list.push_back(tir::transform::Simplify());
      pass_list.push_back(tir::transform::VectorizeLoop(!disable_vectorize));
      pass_list.push_back(tir::transform::InjectVirtualThread());
      pass_list.push_back(tir::transform::StorageRewrite());
      pass_list.push_back(tir::transform::Simplify());
      tvm::Map<String, tvm::PrimExpr> gpu_params{
          {"max_shared_memory_per_block", task->hardware_params->max_shared_memory_per_block},
          {"max_local_memory_per_block", task->hardware_params->max_local_memory_per_block},
          {"max_threads_per_block", task->hardware_params->max_threads_per_block},
          {"max_vector_bytes", task->hardware_params->vector_unit_bytes},
          {"max_vthread", task->hardware_params->max_vthread_extent},
      };
      pass_list.push_back(tir::transform::VerifyGPUCode(gpu_params));
      const auto& optimize = tir::transform::Sequential(pass_list);
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));

    //LOG(INFO) << "optimize";

    const auto& it = mod->functions.find(global_var);
    ICHECK(it != mod->functions.end());
    const auto& prim_func = (*it).second.as<PrimFuncNode>();

    //LOG(INFO) << "node start";

    NodeGather nodeGather;
    nodeGather(prim_func->body);

    //LOG(INFO) << "node";

    for (auto node : nodeGather.node_list) {
      node_list->push_back(node);
    }

    //LOG(INFO) << "edge start";

    EdgeGather edgeGather;
    edgeGather.node_to_index = nodeGather.node_to_index;
    edgeGather(prim_func->body);
    for (auto edge : edgeGather.edge_list) {
      edge_list->push_back(edge);
    }

    //LOG(INFO) << "edge";

  } catch (Error& e) {
    (*error_ct)++;
  }

//  if (i == 0) {
//    std::cout << prim_func->body << std::endl;
//    i++;
//  }

}

void GetGraphFromStates(const Array<State>& states, const std::vector<SearchTask>& tasks,
                        int skip_first_n_feature_extraction, int max_n_bufs,
                        std::vector<std::vector<Node> >* node_list,
                        std::vector<std::vector<Edge> >* edge_list) {
  // extract features
  node_list->assign(states.size(), std::vector<Node>());
  edge_list->assign(states.size(), std::vector<Edge>());

  std::atomic<int> error_ct(0);

  /*
  for (int i = 0; i < states.size(); i++) {
    GetGraph(states[i], tasks[i], max_n_bufs, &(*node_list)[i], 
                                   &(*edge_list)[i], &error_ct);
  }
  */
  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&states, &tasks, &max_n_bufs, &node_list, &edge_list, &error_ct](int i) {
                          GetGraph(states[i], tasks[i], max_n_bufs, &(*node_list)[i], 
                                   &(*edge_list)[i], &error_ct);
                        });
}

void GetGraphFromStates(const Array<State>& states, const SearchTask task,
                        int skip_first_n_feature_extraction, int max_n_bufs,
                        std::vector<std::vector<Node> >* node_list,
                        std::vector<std::vector<Edge> >* edge_list) {
  // extract features
  node_list->assign(states.size(), std::vector<Node>());
  edge_list->assign(states.size(), std::vector<Edge>());

  std::atomic<int> error_ct(0);

  /*
  for (int i = 0; i < states.size(); i++) {
    GetGraph(states[i], task, max_n_bufs, &(*node_list)[i], 
                                   &(*edge_list)[i], &error_ct);
  }
  */
  support::parallel_for(skip_first_n_feature_extraction, states.size(),
                        [&states, &task, &max_n_bufs, &node_list, &edge_list, &error_ct](int i) {
                          GetGraph(states[i], task, max_n_bufs, &(*node_list)[i], 
                                   &(*edge_list)[i], &error_ct);
                        });
}

void GetGraphFromFile(const std::string& filename,
                             int n_lines,
                             int max_n_bufs,
                             std::vector<std::vector<Node> >* node_list,
                             std::vector<std::vector<Edge> >* edge_list,
                             std::vector<float>* normalized_throughputs,
                             std::vector<int>* task_ids) {
  Array<State> states;
  // ArrayNode* pstates = states.CopyOnWrite();
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;
  // task_id -> min_cost
  std::vector<float> min_costs;

  const auto* workload_key_to_tensors =
      tvm::runtime::Registry::Get("auto_scheduler.workload_key_to_tensors");
  ICHECK(workload_key_to_tensors != nullptr);

  // read from file
  RecordReader reader(filename);
  auto cur_inp = make_object<MeasureInputNode>();
  auto cur_res = make_object<MeasureResultNode>();
  while (reader->ReadNext(cur_inp.get(), cur_res.get())) {
    float cost = static_cast<float>(FloatArrayMean(cur_res->costs));
    const std::string& workload_key = cur_inp->task->workload_key;

    SearchTask task;
    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, cur_inp->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      // rebuild task
      Array<te::Tensor> tensors = (*workload_key_to_tensors)(workload_key);
      task = SearchTask(ComputeDAG(tensors), workload_key, cur_inp->task->target,
                        cur_inp->task->target_host, cur_inp->task->hardware_params,
                        cur_inp->task->layout_rewrite_option, cur_inp->task->task_input_names);
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs.push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      min_costs[task_id] = std::min(min_costs[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    // pstates->data.push_back(cur_inp->state);
    states.push_back(cur_inp->state);
    normalized_throughputs->push_back(cost);

    if (n_lines > 0 && static_cast<int>(states.size()) >= n_lines) {
      break;
    }
  }

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = min_costs[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetGraphFromStates(states, tasks, 0, max_n_bufs, node_list, edge_list);
}


void GetGraphFromMeasurePairs(const Array<MeasureInput>& inputs,
                              const Array<MeasureResult>& results,
                              int skip_first_n_feature_extraction, int max_n_bufs,
                              std::vector<std::vector<Node> >* node_list,
                              std::vector<std::vector<Edge> >* edge_list,
                              std::vector<float>* normalized_throughputs,
                              std::vector<int>* task_ids,
                              std::vector<float>* min_costs) {
  Array<State> states;
  std::vector<SearchTask> tasks;

  normalized_throughputs->clear();
  task_ids->clear();
  min_costs->clear();

  // (workload_key, target) -> (search_task, task_id)
  std::unordered_map<std::pair<std::string, std::string>, std::pair<SearchTask, size_t>> task_cache;

  const auto* workload_key_to_tensors =
      tvm::runtime::Registry::Get("auto_scheduler.workload_key_to_tensors");
  ICHECK(workload_key_to_tensors != nullptr);

  //LOG(INFO) << "load func";

  tasks.reserve(inputs.size());
  normalized_throughputs->reserve(inputs.size());
  task_ids->reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    float cost = static_cast<float>(FloatArrayMean(results[i]->costs));
    const std::string& workload_key = inputs[i]->task->workload_key;
    SearchTask task;

    size_t task_id;
    std::pair<std::string, std::string> key(workload_key, inputs[i]->task->target->str());
    auto find_res = task_cache.find(key);
    if (find_res == task_cache.end()) {
      if (inputs[i]->task->compute_dag.defined()) {  // the measure input is complete
        task = inputs[i]->task;
      } else {
        // The measure input is incomplete, rebuild task for incomplete measure pairs read from file
        try {
          Array<te::Tensor> tensors = (*workload_key_to_tensors)(workload_key);
          task =
              SearchTask(ComputeDAG(tensors), workload_key, inputs[i]->task->target,
                         inputs[i]->task->target_host, inputs[i]->task->hardware_params,
                         inputs[i]->task->layout_rewrite_option, inputs[i]->task->task_input_names);
        } catch (std::exception& e) {
          // Cannot build ComputeDAG from workload key, the task may have not been registered in
          // this search round
          continue;
        }
      }
      task_id = task_cache.size();

      // compute min cost for each task
      task_cache.insert(std::make_pair(key, std::make_pair(task, task_id)));
      min_costs->push_back(cost);
    } else {
      std::tie(task, task_id) = find_res->second;
      (*min_costs)[task_id] = std::min((*min_costs)[task_id], cost);
    }

    tasks.push_back(std::move(task));
    task_ids->push_back(task_id);
    states.push_back(inputs[i]->state);
    normalized_throughputs->push_back(cost);
    //LOG(INFO) << "continue";
  }

  //LOG(INFO) << "done";

  for (size_t i = 0; i < normalized_throughputs->size(); ++i) {
    (*normalized_throughputs)[i] = (*min_costs)[(*task_ids)[i]] / (*normalized_throughputs)[i];
  }

  GetGraphFromStates(states, tasks, 0, max_n_bufs, node_list, edge_list);
}



TVM_REGISTER_GLOBAL("auto_scheduler.GetGraphFromStates")
  .set_body([](TVMArgs args, TVMRetValue *ret) {

  Array<State> states = args[0];
  SearchTask task = args[1];
  int max_n_bufs = args[2];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids;
  std::vector<float> min_costs;

  GetGraphFromStates(states, task, 0, max_n_bufs, &node_list, &edge_list);
  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                        std::move(task_ids), std::move(min_costs), &byte_data);

});

TVM_REGISTER_GLOBAL("auto_scheduler.GetGraphFromFile")
.set_body([](TVMArgs args, TVMRetValue *ret) {

  std::string filename = args[0];
  int n_lines = args[1];
  int max_n_bufs = args[2];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids;
  std::vector<float> min_costs;


  GetGraphFromFile(filename, n_lines, max_n_bufs,
      &node_list, &edge_list, &normalized_throughputs, &task_ids);

  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                        std::move(task_ids), std::move(min_costs), &byte_data);
});

TVM_REGISTER_GLOBAL("auto_scheduler.GetGraphFromMeasurePairs")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Array<MeasureInput> inputs = args[0];
  Array<MeasureResult> results = args[1];
  int skip_first_n_feature_extraction = args[2];
  int max_n_bufs = args[3];

  std::vector<std::vector<Node> > node_list;
  std::vector<std::vector<Edge> > edge_list;
  std::vector<float> normalized_throughputs;
  std::vector<int> task_ids; 
  std::vector<float> min_costs;

  GetGraphFromMeasurePairs(inputs, results, skip_first_n_feature_extraction, max_n_bufs,
      &node_list, &edge_list, &normalized_throughputs, &task_ids, &min_costs);

  std::vector<char> byte_data;
  *ret = SerializeGraph(edge_list, node_list, std::move(normalized_throughputs),
                        std::move(task_ids), std::move(min_costs), &byte_data);
});


}   // namespace auto_scheduler
}   // namespace tvm
